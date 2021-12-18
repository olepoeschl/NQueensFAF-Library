package de.nqueensfaf;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import de.nqueensfaf.util.OnProgressUpdateCallback;
import de.nqueensfaf.util.OnTimeUpdateCallback;

/**
 * <p>
 * Abstract class for implementing Solvers.
 * </p>
 * <p>
 * To implement an own solver for the n queens problem, simply create a class that extends this one. 
 * Then the solver will automatically have the util functions like adding callbacks etc. and the given basic solver class structure makes it easier.
 * </p>
 * 
 * @author olepo
 */
public abstract class Solver {

	/**
	 * board size
	 */
	protected int N;
	/**
	 * callback that is executed on every time update
	 */
	protected OnTimeUpdateCallback  onTimeUpdateCallback = (duration) -> {};
	/**
	 * callback that is executed on every progress update
	 */
	protected OnProgressUpdateCallback  onProgressUpdateCallback = (progress, solutions) -> {};
	/**
	 * delay between time updates
	 */
	protected long timeUpdateDelay = NQueensFAF.DEFAULT_TIME_UPDATE_DELAY;
	/**
	 * delay between progress updates
	 */
	protected long progressUpdateDelay = NQueensFAF.DEFAULT_PROGRESS_UPDATE_DELAY;
	/**
	 * executor of the update callbacks (uc)
	 * @see #startUpdateCallerThreads()
	 */
	private ThreadPoolExecutor ucExecutor;
	/**
	 * list of callbacks to be executed before the run() function of the solver is called
	 * @see #solve()
	 */
	private ArrayList<Runnable> initialization = new ArrayList<Runnable>();
	/**
	 * list of callbacks to be executed after the run() function of the solver is finished
	 * @see #solve()
	 */
	private ArrayList<Runnable> termination = new ArrayList<Runnable>();
	/**
	 * current state of the {@link Solver}
	 * @see NQueensFAF
	 */
	private int state = NQueensFAF.IDLE;
	/**
	 * thread that executes the solvers run() function if {@link #solveAsync()} is called
	 */
	private Thread t;
	/**
	 * if true, makes the Solver not calling the onProgressUpdate and onTimeUpdate callbacks
	 */
	private boolean progressUpdatesEnabled = true;
	/**
	 * thread for the {@link #autoSave} function
	 * @see 
	 */
	private Thread autoSaverThread;
	/**
	 * if true, the Solver automatically deletes a file created by the auto save function using
	 * only used when {@link #autoSave} is true
	 */
	private boolean autoDeleteEnabled = false;
	/**
	 * if true, the Solver automatically calls store() using {@link #autoSaveFilename} if a specific progress percentage step is completed
	 * @see #autoSavePercentageStep
	 */
	private boolean autoSaveEnabled = false;
	/**
	 * after the progress increases by this or more percentage, store() is automatically called.
	 * only used when {@link #autoSave} is true
	 */
	private int autoSavePercentageStep = 10;
	/**
	 * the filename (format) used for the autosave feature.
	 * only used when {@link #autoSave} is true
	 * @see #autoSavePercentageStep
	 */
	private String autoSaveFilename = "N{N}.nqf";
	/**
	 * for controlflow. Avoids checkForPreparation() being called twice in case solveAsync() is used.
	 */
	private boolean preparationChecked = false;
	/**
	 * number of solutions - only used by the method that solves the problem for small N's.
	 * @see #solveSmallBoard()
	 * @see #nq(int, int, int, int, int, int)
	 */
	int solutionsSmallN = 0;

	// abstract methods
	/**
	 * Solves the N-Queens Problem.
	 */
	protected abstract void run();
	/**
	 * Saves the current progress of the {@link Solver} in a file so that it can be continued at some time later.
	 * @param filepath path/name of the file the progress should be written in (existent or non existent)
	 * @throws IOException
	 */
	protected abstract void store_(String filepath) throws IOException;
	/**
	 * Reads the progress of an old run of the {@link Solver} and restores this state so that it can be continued.
	 * @param filepath path/name of the file the progress was written in
	 * @throws IOException
	 * @throws ClassNotFoundException 
	 */
	protected abstract void restore_(String filepath) throws IOException, ClassNotFoundException, ClassCastException;
	/**
	 * States if the Solver has been restored and therefore contains restored values.
	 * @return true if restore() was called and the Solver was not started or resetted since, otherwise false
	 */
	public abstract boolean isRestored();
	/**
	 * Resets the {@link Solver}; therefore, applies default values to all Solver state and progress related variables.
	 */
	public abstract void reset();
	/**
	 * Gets the total runtime of the {@link Solver} in the current run.
	 * @return total runtime of the current run or 0 if the Solver is currently not running
	 */
	public abstract long getDuration();
	/**
	 * Gets the percentage of the {@link Solver}'s task that is completed.
	 * @return percentage of the {@link Solver}'s task that is completed
	 */
	public abstract float getProgress();
	/**
	 * Gets the current count of found solutions.
	 * @return current count of found solutions
	 */
	public abstract long getSolutions();

	/**
	 * Calls all initialization callbacks, then starts the {@link Solver}'s run()-method, then calls all termination callbacks.
	 * @throws InterruptedException if the time limit is exceeded while waiting for the Solverthread to shutdown after terminating (time limit is the bigger one of progressUpdateDelay and timeUpdateDelay)
	 * @see #checkForPrepation()
	 * @see #run()
	 */
	public final void solve() {
		if(!preparationChecked)
			checkForPreparation();
		state = NQueensFAF.INITIALIZING;
		initializationCaller();
		ucExecutor = (ThreadPoolExecutor) Executors.newFixedThreadPool(2);

		state = NQueensFAF.RUNNING;
		startUpdateCallerThreads();
		if(autoSaveEnabled)
			startAutoSaverThread();
		run();

		state = NQueensFAF.TERMINATING;
		terminationCaller();
		ucExecutor.shutdown();
		try {
			ucExecutor.awaitTermination(timeUpdateDelay > progressUpdateDelay ? timeUpdateDelay+1000 : progressUpdateDelay+1000, TimeUnit.MILLISECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		state = NQueensFAF.IDLE;
	}
	/**
	 * Asynchronously starts the {@link Solver} using a new thread and then waits till the Solver is successfully started.
	 * @throws {@link IllegalStateException} if the {@link Solver} is already running
	 * @see #solve()
	 */
	public final void solveAsync() {
		checkForPreparation();
		if(!isIdle()) {
			throw new IllegalStateException("Solver is already started");
		}
		t = new Thread(() -> solve());
		t.start();
	}
	/**
	 * Waits for the asynchronously running {@link Solver} to finish.
	 * @throws {@link InterruptedException} if the {@link Solver} is not running
	 * @see #solveAsync()
	 */
	public final void waitFor() throws InterruptedException {
		if(t == null) {
			return;
		}
		t.join();
	}
	/**
	 * Checks if all prepartions are done correctly.
	 * * @throws {@link IllegalStateException} if N equals 0, the {@link Solver} is already running or if getProgress() >= 100
	 */
	private void checkForPreparation() {
		if(N == 0) {
			state = NQueensFAF.IDLE;
			throw new IllegalStateException("Board size was not set");
		}
		if(!isIdle()) {
			state = NQueensFAF.IDLE;
			throw new IllegalStateException("Solver is already started");
		}
		if(getProgress() == 1.0f) {
			state = NQueensFAF.IDLE;
			throw new IllegalStateException("Solver is already done, nothing to do here");
		}
	}

	/**
	 * Starts the threads that continously update the {@link Solver}'s duration and progress using the related delay.
	 * The threads run until the {@link Solver} is finished.
	 * @see #timeUpdateDelay
	 * @see #progressUpdateDelay
	 */
	private void startUpdateCallerThreads() {
		if(onTimeUpdateCallback != null) {
			ucExecutor.submit(() -> {
				long tmpTime = 0;
				while(isRunning()) {
					if(getDuration() != tmpTime) {
						onTimeUpdateCallback.onTimeUpdate(getDuration());
						tmpTime = getDuration();
					}
					if(!isRunning())
						break;
					try {
						Thread.sleep(timeUpdateDelay);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
				onTimeUpdateCallback.onTimeUpdate(getDuration());
			});
		}
		if(progressUpdatesEnabled && onProgressUpdateCallback != null) {
			ucExecutor.submit(() -> {
				float tmpProgress = 0;
				long tmpSolutions = 0;
				while(isRunning()) {
					if(getProgress() != tmpProgress || getSolutions() != tmpSolutions) {
						onProgressUpdateCallback.onProgressUpdate(getProgress(), getSolutions());
						tmpProgress = getProgress();
						tmpSolutions = getSolutions();
					}
					if(!isRunning())
						break;
					try {
						Thread.sleep(progressUpdateDelay);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
				onProgressUpdateCallback.onProgressUpdate(getProgress(), getSolutions());
			});
		}
	}

	/**
	 * Starts the thread that continously checks if the progress has increased by {@link #autoSavePercentageStep} or more,
	 * and if yes, calls store() using {@link #autoSaveFilename} the current Solver.
	 * Only used when {@link #autoSave} is true
	 * @see #autoSaverThread
	 */
	private void startAutoSaverThread() {
		autoSaverThread = new Thread(() -> {
			String filename = autoSaveFilename;
			filename = filename.replaceAll("#N#", ""+ N);
			if(!filename.endsWith(".faf")) {
				filename += ".faf";
			}
			float tmpProgress = 0;
			while(isRunning()) {
				if(getProgress()*100 >= tmpProgress + autoSavePercentageStep) {
					try {
						store(filename, false);
					} catch (IOException e) {
						e.printStackTrace();
					}
					tmpProgress = getProgress();
				}
				try {
					Thread.sleep(300);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			if(autoDeleteEnabled) {
				try {
					new File(filename).delete();
				} catch(SecurityException e) {}
			}
		});
		autoSaverThread.start();
	}
	
	/**
	 * Can be used for solving the problem for small N's (board sizes).
	 * @return the number of solutions
	 */
	protected int solveSmallBoard() {
		solutionsSmallN = 0;
		int mask = (1 << getN()) - 1;
		nq(0, 0, 0, 0, mask, mask);
		return solutionsSmallN;
	}
	/**
	 * Solves the N-Queens-problem in a very simple and easy way.
	 * Only for small N's, bigger N's would take much longer with this method.
	 * @param ld soon to read about in an extern documentation paper
	 * @param rd soon to read about in an extern documentation paper
	 * @param col soon to read about in an extern documentation paper
	 * @param row soon to read about in an extern documentation paper
	 * @param free soon to read about in an extern documentation paper
	 * @param mask soon to read about in an extern documentation paper
	 * @param solutions soon to read about in an extern documentation paper
	 */
	private void nq(int ld, int rd, int col, int row, int free, int mask) {
		if(row == getN()-1) {
			solutionsSmallN++;
			return;
		}

		int bit;
		int nextfree;

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~((ld|bit)<<1 | (rd|bit)>>1 | col|bit) & mask;

			if(nextfree > 0)
				nq((ld|bit)<<1, (rd|bit)>>1, col|bit, row+1, nextfree, mask);
		}
	}

	/**
	 * Replaces all invalid characters of a String that is supposed to be a filename.
	 * @param filename name of the file the Solver should store its progress state in
	 * @return the given filename but without invalid characters reagrding filenames
	 */
	private String getValidFilename(String filename) throws IllegalArgumentException {
		String newFilename = filename.replace("^\\.+", "").replaceAll("[\\\\/:*?\"<>|]", "");
		if(newFilename.length() == 0) {
			throw new IllegalArgumentException("Invalid filename: '" + filename + "'");
		}
		return newFilename;
	}
	
	/**
	 * Wraps {@link #store_(String)}.
	 * @param filepath path/name of the file the Solver's progress state should be stored in
	 * @param bypassValidityCheck if true, does not check if the given filename is valid; should be true for absolute paths and false for filenames
	 * @throws IOException
	 * @throws IllegalArgumentException
	 */
	public void store(String filepath, boolean bypassValidityCheck) throws IOException, IllegalArgumentException {
		if(!bypassValidityCheck)
			filepath = getValidFilename(filepath);
		store_(filepath);
	}

	/**
	 * Wraps {@link #restore_(String)}.
	 * @param filepath path/name of the file the Solver's progress state should be restored from
	 * @throws IOException 
	 * @throws ClassCastException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalArgumentException
	 */
	public void restore(String filepath) throws IOException, ClassNotFoundException, ClassCastException, IllegalArgumentException {
		restore_(filepath);
	}
	
	/**
	 * Adds a callback that will be executed on start of the {@link Solver}.
	 * The callbacks will be called in reversed insertion order.
	 * Chainable.
	 * @param r the callback as {@link Runnable}
	 * @return the {@link Solver}
	 * @throws {@link IllegalArgumentException} if r is null
	 * @see #solve()
	 */
	public final Solver addInitializationCallback(Runnable r) {
		if(r == null) {
			throw new IllegalArgumentException("initializationCallback must not be null");
		}
		initialization.add(r);
		return this;
	}

	/**
	 * Adds a callback that will be executed on finish of the {@link Solver}.
	 * The callbacks will be called in reversed insertion order.
	 * Chainable.
	 * @param r the callback as {@link Runnable}
	 * @return the {@link Solver}
	 * @throws {@link IllegalArgumentException} if r is null
	 * @see #solve()
	 */
	public final Solver addTerminationCallback(Runnable r) {
		if(r == null) {
			throw new IllegalArgumentException("terminationCallback must not be null");
		}
		termination.add(r);
		return this;
	}

	/**
	 * Calls all initialization callbacks in reversed insertion order.
	 */
	private void initializationCaller() {
		for(Runnable r : initialization) {
			r.run();
		}
	}

	/**
	 * Calls all termination callbacks in reversed insertion order.
	 */
	private void terminationCaller() {
		for(Runnable r : termination) {
			r.run();
		}
	}

	// Getters and Setters
	/**
	 * Gets {@link #N}.
	 * @return {@link #N}
	 */
	public final int getN() {
		return N;
	}
	/**
	 * Sets {@link #N}.
	 * Chainable.
	 * @param n boardsize 
	 * @return the {@link Solver}
	 * @throws {@link IllegalStateException} if the {@link Solver} is already running
	 * @throws {@link IllegalArgumentException} if the boardsize is invalid 
	 */
	public final Solver setN(int n) {
		if(!isIdle()) {
			throw new IllegalStateException("Cannot set board size while solving");
		}
		if(n <= 0 || n > 31) {
			throw new IllegalArgumentException("Board size must be a number between 0 and 32 (not inclusive)");
		}
		N = n;
		return this;
	}

	/**
	 * Gets {@link #onTimeUpdateCallback}.
	 * @return {@link #onTimeUpdateCallback}
	 */
	public final OnTimeUpdateCallback getOnTimeUpdateCallback() {
		return onTimeUpdateCallback;
	}
	/**
	 * Sets {@link #onTimeUpdateCallback} or sets a void callback if the parameter is null.
	 * Chainable.
	 * @param onTimeUpdateCallback callback to be called on each time update
	 * @return the {@link Solver}
	 */
	public final Solver setOnTimeUpdateCallback(OnTimeUpdateCallback onTimeUpdateCallback) {
		if(onTimeUpdateCallback == null) {
			this.onTimeUpdateCallback = (duration) -> {};
		} else {
			this.onTimeUpdateCallback = onTimeUpdateCallback;
		}
		return this;
	}

	/**
	 * Gets {@link #onProgressUpdateCallback}.
	 * @return {@link #onProgressUpdateCallback}
	 */
	public final OnProgressUpdateCallback getOnProgressUpdateCallback() {
		return onProgressUpdateCallback;
	}
	/**
	 * Sets {@link #onProgressUpdateCallback} or sets a void callback if the parameter is null.
	 * Chainable.
	 * @param onProgressUpdateCallback callback to be called on each progress update
	 * @return the {@link Solver}
	 */
	public final Solver setOnProgressUpdateCallback(OnProgressUpdateCallback onProgressUpdateCallback) {
		if(onProgressUpdateCallback == null) {
			this.onProgressUpdateCallback = (progress, solutions) -> {};
		} else {
			this.onProgressUpdateCallback = onProgressUpdateCallback;
		}
		return this;
	}

	/**
	 * Gets {@link #timeUpdateDelay}.
	 * @return {@link #timeUpdateDelay}
	 */
	public final long getTimeUpdateDelay() {
		return timeUpdateDelay;
	}
	/**
	 * Sets {@link #timeUpdateDelay}.
	 * Chainable.
	 * @param timeUpdateDelay
	 * @return the {@link Solver}
	 * @throws {@link IllegalArgumentException} if the given delay is <= 0
	 */
	public final Solver setTimeUpdateDelay(long timeUpdateDelay) {
		if(timeUpdateDelay < 0) {
			throw new IllegalArgumentException("timeUpdateDelay must be a number >= 0");
		} else if(timeUpdateDelay== 0) {
			this.timeUpdateDelay = NQueensFAF.DEFAULT_TIME_UPDATE_DELAY;
			return this;
		}
		this.timeUpdateDelay = timeUpdateDelay;
		return this;
	}

	/**
	 * Gets {@link #progressUpdateDelay}.
	 * @return {@link #progressUpdateDelay}
	 */
	public final long getProgressUpdateDelay() {
		return progressUpdateDelay;
	}
	/**
	 * Sets {@link #progressUpdateDelay}.
	 * Chainable.
	 * @param progressUpdateDelay
	 * @return the {@link Solver}
	 * @throws {@link IllegalArgumentException} if the given delay is <= 0
	 */
	public final Solver setProgressUpdateDelay(long progressUpdateDelay) {
		if(progressUpdateDelay < 0) {
			throw new IllegalArgumentException("progressUpdateDelay must be a number >= 0");
		} else if(progressUpdateDelay== 0) {
			this.progressUpdateDelay = NQueensFAF.DEFAULT_PROGRESS_UPDATE_DELAY;
			return this;
		}
		this.progressUpdateDelay = progressUpdateDelay;
		return this;
	}

	/**
	 * Gets {@link #progressUpdatesEnabled}.
	 * @return {@link #progressUpdatesEnabled}
	 */
	public final boolean areProgressUpdatesEnabled() {
		return progressUpdatesEnabled;
	}
	/**
	 * Enables or disables progress updates.
	 * Chainable.
	 * @param progressUpdatesEnabled if true, enables updates (default value); if false, disables updates.
	 * @return the {@link Solver}
	 */
	public final Solver setProgressUpdatesEnabled(boolean progressUpdatesEnabled) {
		this.progressUpdatesEnabled = progressUpdatesEnabled;
		return this;
	}

	/**
	 * Gets {@link #autoSaveEnabled}.
	 * @return {@link #autoSaveEnabled}
	 */
	public final boolean isAutoSaveEnabled() {
		return autoSaveEnabled;
	}
	/**
	 * Enables or disables the auto save function.
	 * Chainable.
	 * @param autoSaveEnabled if true, enables automatic saving of the current Solver; if false, disables automatic saving (default value).
	 * @return the {@link Solver}
	 */
	public final Solver setAutoSaveEnabled(boolean autoSaveEnabled) {
		this.autoSaveEnabled = autoSaveEnabled;
		return this;
	}

	/**
	 * Gets {@link #autoDeleteEnabled}.
	 * @return {@link #autoDeleteEnabled}
	 */
	public final boolean isAutoDeleteEnabled() {
		return autoDeleteEnabled;
	}
	/**
	 * Enables or disables the auto delete function.
	 * Chainable.
	 * @param autoDeleteEnabled if true, enables automatic deleting of the files created by the auto save function
	 * @return the {@link Solver}
	 */
	public final Solver setAutoDeleteEnabled(boolean autoDeleteEnabled) {
		this.autoDeleteEnabled = autoDeleteEnabled;
		return this;
	}
	
	/**
	 * Gets {@link #autoSavePercentageStep}.
	 * @return {@link #autoSavePercentageStep}
	 */
	public final int getAutoSavePercentageStep() {
		return autoSavePercentageStep;
	}
	/**
	 * Sets {@link #autoSavePercentageStep}.
	 * Chainable.
	 * @param autoSavePercentageStep
	 * @return the {@link Solver}
	 */
	public final Solver setAutoSavePercentageStep(int autoSavePercentageStep) {
		if(autoSavePercentageStep <= 0 || autoSavePercentageStep >= 100) {
			throw new IllegalArgumentException("progressUpdateDelay must be a number between 0 and 100");
		}
		this.autoSavePercentageStep = autoSavePercentageStep;
		return this;
	}
	
	/**
	 * Gets {@link #autoSaveFilename}.
	 * @return {@link #autoSaveFilename}
	 */
	public final String getAutoSaveFilename() {
		return autoSaveFilename;
	}
	/**
	 * Sets {@link #autoSaveFilename}.
	 * Chainable.
	 * @param autoSaveFilename
	 * @return the {@link Solver}
	 */
	public final Solver setAutoSaveFilename(String autoSaveFilename) {
		this.autoSaveFilename = autoSaveFilename;
		return this;
	}

	/**
	 * Returns true if the {@link Solver}'s state is {@link NQueensFAF#IDLE}
	 * @return true if the current state of the {@link Solver} is {@link NQueensFAF#IDLE}
	 */
	public final boolean isIdle() {
		return state == NQueensFAF.IDLE;
	}
	/**
	 * Returns true if the {@link Solver}'s state is {@link NQueensFAF#INITIALIZING}
	 * @return true if the current state of the {@link Solver} is {@link NQueensFAF#INITIALIZING}
	 */
	public final boolean isInitializing() {
		return state == NQueensFAF.INITIALIZING;
	}
	/**
	 * Returns true if the {@link Solver}'s state is {@link NQueensFAF#RUNNING}
	 * @return true if the current state of the {@link Solver} is {@link NQueensFAF#RUNNING}
	 */
	public final boolean isRunning() {
		return state == NQueensFAF.RUNNING;
	}
	/**
	 * Returns true if the {@link Solver}'s state is {@link NQueensFAF#TERMINATING}
	 * @return true if the current state of the {@link Solver} is {@link NQueensFAF#TERMINATING}
	 */
	public final boolean isTerminating() {
		return state == NQueensFAF.TERMINATING;
	}
}
