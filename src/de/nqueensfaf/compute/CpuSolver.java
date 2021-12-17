package de.nqueensfaf.compute;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import de.nqueensfaf.Solver;

public class CpuSolver extends Solver {

	private final int smallestN = 6;
	private int threadcount = 1;
	private long start, end;
	private HashSet<Integer> 
		startConstellations = new HashSet<Integer>();
	private ArrayList<CpuSolverThread> threads = new ArrayList<CpuSolverThread>();
	private int startConstCount, solvedConstellations;
	private long timePassed = 0, pauseStart = 0;
	private long solutions;
	private boolean restored = false;
	private ArrayList<Runnable> pausing = new ArrayList<Runnable>();
	
	// inherited functions
	@Override
	protected void run() {
		// check if run is called without calling reset after a run call had finished
		if(start != 0) {
			throw new IllegalStateException("You first have to call reset() when calling solve() multiple times on the same object");
		}
		
		start = System.currentTimeMillis();
		if(N <= smallestN) {	// if N is very small, use the simple Solver from the parent class
			solutions = solveSmallBoard();
			end = System.currentTimeMillis();
			// simulate progress = 100
			startConstCount = 1;
			solvedConstellations = 1;
			return;
		}
		if(!restored) {
			genConstellations();
			startConstCount = startConstellations.size();
		}
		
		// split starting constellations in [cpu] many lists (splitting the work for the threads)
		ArrayList<ArrayDeque<Integer>> threadConstellations = new ArrayList<ArrayDeque<Integer>>(threadcount);
		for(int i = 0; i < threadcount; i++) {
			threadConstellations.add(new ArrayDeque<Integer>());
		}
		int i = 0;
		for(int constellation : startConstellations) {
			threadConstellations.get((i++) % threadcount).addFirst(constellation);
		}

		// start the threads and wait until they are all finished
		ExecutorService executor = Executors.newFixedThreadPool(threadcount);
		for(i = 0; i < threadcount; i++) {
			CpuSolverThread cpuSolverThread = new CpuSolverThread(N, threadConstellations.get(i), this);
			threads.add(cpuSolverThread);
			executor.submit(cpuSolverThread);
		}
		
		// wait for the threads to finish
		executor.shutdown();
		try {
			if(executor.awaitTermination(365, TimeUnit.DAYS)) {
				end = System.currentTimeMillis();
				// done 
			} else {
				end = System.currentTimeMillis();
				// not done
			}
		} catch (InterruptedException e1) {
			e1.printStackTrace();
		}
		restored = false;
	}
	
	@Override
	public void store_(String filepath) throws IOException {
		// if Solver was not even started yet, throw exception
		if(start == 0) {
			throw new IllegalStateException("Nothing to be saved");
		}
		// get current progress values
		HashSet<Integer> startConstellations = new HashSet<Integer>();
		for(CpuSolverThread t : threads) {
			startConstellations.addAll(t.getRemainingConstellations());
		}
		long solutions = this.solutions;
		for(CpuSolverThread t : threads) {
			solutions += t.getSolutions();
		}
		long timePassed = getDuration();
		RestorationInformation resInfo = new RestorationInformation(N, startConstellations, timePassed, solutions, startConstCount);
		
		FileOutputStream fos = new FileOutputStream(filepath);
		ObjectOutputStream oos = new ObjectOutputStream(fos);
		oos.writeObject(resInfo);
		oos.flush();
		oos.close();
		fos.close();
	}

	@Override
	public void restore_(String filepath) throws IOException, ClassNotFoundException, ClassCastException {
		if(!isIdle()) {
			throw new IllegalStateException("Cannot restore while the Solver is running");
		}
		RestorationInformation resInfo;
		FileInputStream fis = new FileInputStream(filepath);
		ObjectInputStream ois = new ObjectInputStream(fis);
		resInfo = (RestorationInformation) ois.readObject();
		ois.close();
		fis.close();

		reset();
		N = resInfo.N;
		startConstellations = resInfo.startConstellations;
		timePassed = resInfo.timePassed;
		solutions = resInfo.solutions;
		startConstCount = resInfo.startConstCount;
		solvedConstellations = startConstCount - startConstellations.size();
		restored = true;
	}
	
	@Override
	public boolean isRestored() {
		return restored;
	}
	
	@Override
	public void reset() {
		start = 0;
		end = 0;
		timePassed = 0;
		pauseStart = 0;
		startConstellations.clear();
		solvedConstellations = 0;
		solutions = 0;
		threads.clear();
		startConstCount = 0;
		restored = false;
		System.gc();
	}

	@Override
	public long getDuration() {
		if(isRunning()) {
			if(isPaused()) {
				return timePassed;
			} else {
				return timePassed + System.currentTimeMillis() - start;
			}
		} else {
			if(isPaused()) {
				return timePassed;
			} else {
				return timePassed + end - start;
			}
		}
	}

	@Override
	public float getProgress() {
		if(restored && isIdle())
			return (float) solvedConstellations / startConstCount;
		float done = solvedConstellations;
		for(CpuSolverThread t : threads) {
			done += t.getDone();
		}
		return done / startConstCount;
	}

	@Override
	public long getSolutions() {
		long solutions = this.solutions;
		for(CpuSolverThread t : threads) {
			solutions += t.getSolutions();
		}
		return solutions;
	}
	
	// own functions
	private void genConstellations() {
		startConstellations.clear();

		// halfN half of N rounded up
		final int halfN = (N + 1) / 2;

		// calculating start constellations with the first Queen on square (0,0)
		for(int j = 1; j < N-2; j++) {						// j is idx of Queen in last row				
			for(int l = j+1; l < N-1; l++) {				// l is idx of Queen in last col
				startConstellations.add(toijkl(0, j, 0, l));
			}
		}

		// calculate starting constellations for no Queens in corners
		for(int k = 1; k < halfN; k++) {						// go through first col
			for(int l = k+1; l < N-1; l++) {					// go through last col
				for(int i = k+1; i < N-1; i++) {				// go through first row
					if(i == N-1-l)								// skip if occupied
						continue;
					for(int j = N-k-2; j > 0; j--) {			// go through last row
						if(j==i || l == j)
							continue;

						if(!checkRotations(i, j, k, l)) {		// if no rotation-symmetric starting constellation already found
							startConstellations.add(toijkl(i, j, k, l));
						}
					}
				}
			}
		}
	}
	
	// true, if starting constellation rotated by any angle has already been found
	private boolean checkRotations(int i, int j, int k, int l) {
		// rot90
		if(startConstellations.contains(((N-1-k)<<24) + ((N-1-l)<<16) + (j<<8) + i)) 
			return true;

		// rot180
		if(startConstellations.contains(((N-1-j)<<24) + ((N-1-i)<<16) + ((N-1-l)<<8) + N-1-k)) 
			return true;

		// rot270
		if(startConstellations.contains((l<<24) + (k<<16) + ((N-1-i)<<8) + N-1-j)) 
			return true;

		return false;
	}

	// wrap i, j, k and l to one integer using bitwise movement
	private int toijkl(int i, int j, int k, int l) {
		return (i<<24) + (j<<16) + (k<<8) + l;
	}

	// for user interaction
	public void pause() {
		if(!isRunning()) {
			throw new IllegalStateException("unable to pause a CpuSolver when it is not running");
		}
		if(isPaused()) {
			throw new IllegalStateException("unable to pause a CpuSolver when it is already paused");
		}
		for(CpuSolverThread t : threads) {
			t.pauseThread();
		}
	}
	
	public void cancel() {
		if(!isRunning()) {
			throw new IllegalStateException("unable to cancel a CpuSolver when it is not running");
		}
		for(CpuSolverThread t : threads) {
			t.cancelThread();
		}
	}
	
	public void resume() {
		if(!isRunning()) {
			throw new IllegalStateException("unable to resume a CpuSolver when it is not running");
		}
		boolean paused = isPaused();
		for(CpuSolverThread t : threads) {
			t.resumeThread();
		}
		if(paused) {
			start = System.currentTimeMillis();
			pauseStart = 0;
		}
	}
	
	public boolean isPaused() {
		if(threads.size() <= 0) {
			return false;
		}
		for(CpuSolverThread t : threads) {
			if(!t.isPaused())
				return false;
		}
		return true;
	}
	
	public boolean wasCanceled() {
		if(isRunning() || N <= smallestN) {
			return false;
		}
		for(CpuSolverThread t : threads) {
			if(!t.wasCanceled())
				return false;
		}
		return true;
	}
	
	public void addOnPauseCallback(Runnable r) {
		if(r == null) {
			throw new IllegalArgumentException("pausing callback must not be null");
		}
		pausing.add(r);
	}
	
	// is called from the SolverThreads to measure correct time of the start of the pause 
	synchronized void onPauseStart() {
		if(isPaused() && pauseStart == 0) {
			long now = System.currentTimeMillis();
			timePassed += now - start;
			pauseStart = now;
			for(Runnable r : pausing) {
				r.run();
			}
		}
	}
	
	// getters and setters
	public void setThreadcount(int threadcount) {
		if(threadcount < 1 || threadcount > Runtime.getRuntime().availableProcessors()) {
			throw new IllegalArgumentException("threadcount must be a number between 1 and " + Runtime.getRuntime().availableProcessors() + " (=your CPU's number of logical cores) (inclusive)");
		}
		this.threadcount = threadcount;
	}
	
	public int getThreadcount() {
		return threadcount;
	}
	
	// for saving and restoring
	private record RestorationInformation(int N, HashSet<Integer> startConstellations, long timePassed, long solutions, int startConstCount) implements Serializable {
		RestorationInformation(int N, HashSet<Integer> startConstellations, long timePassed, long solutions, int startConstCount) {
			this.N = N;
			this.startConstellations = startConstellations;
			this.timePassed = timePassed;
			this.solutions = solutions;
			this.startConstCount = startConstCount;
		}
	}
}
