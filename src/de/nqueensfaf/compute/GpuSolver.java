package de.nqueensfaf.compute;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.lwjgl.BufferUtils;
import org.lwjgl.LWJGLException;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CL10;
import org.lwjgl.opencl.CLCommandQueue;
import org.lwjgl.opencl.CLContext;
import org.lwjgl.opencl.CLDevice;
import org.lwjgl.opencl.CLEvent;
import org.lwjgl.opencl.CLKernel;
import org.lwjgl.opencl.CLMem;
import org.lwjgl.opencl.CLPlatform;
import org.lwjgl.opencl.CLProgram;
import org.lwjgl.opencl.Util;

import de.nqueensfaf.NQueensFAF;
import de.nqueensfaf.Solver;

public class GpuSolver extends Solver {

	private static Path tempDir;
	private static boolean openclable = true;
	// check if a OpenCL-capable device is available and block GpuSolver and print an error message, if not
	static {
		int checked = checkOpenCL();
		switch(checked) {
		case 0:
			openclable = false;
			break;
		case 1:
			openclable = true;
			break;
		case -1:
			// checking for OpenCL-capable devices was not possible
			if(NQueensFAF.getIgnoreOpenCLCheck()) {
				openclable = true;
				break;
			}
			System.err.println("Unable to check for OpenCL-capable devices.");
			System.err.println("To get rid of this warning, install 'clinfo' (better option) or use NQueensFAF.setIgnoreOpenCLCheck(true) (will crash the JVM if no OpenCL-capable device is found).");
			openclable = false;
			break;
		}
		if(!openclable) {
			System.err.println("No OpenCL-capable device was found. GpuSolver is not available.");
		} else {
			// enables the easy use of lwjgl out of the jar-archive. The customer only need the lwjgl.jar. (LWJGL 2.9.3)
			loadLWJGLNative();
			// initialize OpenCL
			try {
				CL.create();
			} catch (LWJGLException e) {
				e.printStackTrace();
			}
			// add shutdown hook that destroys OpenCL when the program is exited
			Runtime.getRuntime().addShutdownHook(new Thread(() -> {
				CL.destroy();
			}));
		}
		if(tempDir != null) {
			// add shutdown hook to delete the created temporary directory
			Runtime.getRuntime().addShutdownHook(new Thread(() -> {
				deleteTempDir();
			}));
		}
	}

	// OpenCL stuff
	private IntBuffer errBuf, resBuf, progressBuf;
	private CLContext context;
	private CLPlatform platform;
	private List<CLDevice> devices;
	private CLDevice device;
	private CLCommandQueue xqueue, memqueue;
	private CLProgram program;
	private CLKernel kernel;
	private CLMem ldMem, rdMem, colMem, startjklMem, resMem, progressMem;
	private int WORKGROUP_SIZE = 64;
	private int WORKGROUP_FILL_MIN = (int) ((1-0.15625) * WORKGROUP_SIZE) + 5; // 22 if WOKRGROUP_SIZE is 32, 49 if it is 64
	private int globalWorkSize;

	// calculation related stuff
	private GpuConstellationsGenerator generator;
	private int startConstCount;
	private ArrayList<Integer> ldList, rdList, colList, symList, startjklList;
	private long solutions, savedSolutions;
	private long duration, start, end, savedDuration;
	private float progress;
	private int savedSolvedConstellations;
	
	// control flow variables
	private boolean gpuDone = false;
	private boolean restored = false;
	
	public GpuSolver() {
		if(openclable)
			getAvailableDevices();		// fill the devices list with all available devices
	}
	
	// inherited functions
	@Override
	protected void run() {
		if(start != 0) {
			throw new IllegalStateException("You first have to call reset() when calling solve() multiple times on the same object");
		}
		if(!openclable) {
			throw new IllegalStateException("No OpenCL-capable device was found. GpuSolver is not available.");
		}
		if(N <= 6) {	// if N is very small, use the simple Solver from the parent class
			// prepare simulating progress = 100
			progress = 1f;
			gpuDone = true;
			start = System.currentTimeMillis();
			solutions = solveSmallBoard();
			end = System.currentTimeMillis();
			// simulate progress = 100
			startConstCount = 1;
			return;
		}
		if(device == null) {
			throw new IllegalStateException("You have to choose a device by calling setDevice() before starting the Solver. See all available devices using getAvailableDevices()");
		}
		// if init fails, do not proceed
		try {
			init();
		} catch (LWJGLException e) {
			e.printStackTrace();
			return;
		}
		transferDataToDevice();
		explosionBoost9000();
		readResults();
		terminate();
		restored = false;
	}

	@Override
	public void store_(String filepath) throws IOException {
		// if Solver was not even started yet, throw exception
		if(start == 0 || getProgress() >= 1f) {
			throw new IllegalStateException("Nothing to be saved");
		}
		long solutions = savedSolutions;
		ArrayList<Integer> 
				ldListTmp = new ArrayList<Integer>(),
				rdListTmp = new ArrayList<Integer>(),
				colListTmp = new ArrayList<Integer>(),
				startjklListTmp = new ArrayList<Integer>(),
				symListTmp = new ArrayList<Integer>();
		synchronized(resMem) {
			synchronized(progressMem) {
				CL10.clEnqueueReadBuffer(memqueue, resMem, CL10.CL_TRUE, 0, resBuf, null, null);
				CL10.clEnqueueReadBuffer(memqueue, progressMem, CL10.CL_TRUE, 0, progressBuf, null, null);
				for(int i = 0; i < globalWorkSize; i++) {
					if(progressBuf.get(i) == 1) {
						solutions += resBuf.get(i) * symList.get(i);
					} else if(progressBuf.get(i) == 0 && startjklList.get(i) != (69 << 15)) {
						ldListTmp.add(ldList.get(i));
						rdListTmp.add(rdList.get(i));
						colListTmp.add(colList.get(i));
						startjklListTmp.add(startjklList.get(i));
						symListTmp.add(symList.get(i));
					}
				}
			}
		}
		RestorationInformation resInfo = new RestorationInformation(N, getDuration(), solutions, startConstCount, ldListTmp, rdListTmp, colListTmp, startjklListTmp, symListTmp);

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
		savedDuration = resInfo.duration;
		savedSolutions = resInfo.solutions;
		startConstCount = resInfo.startConstCount;
		savedSolvedConstellations = startConstCount - resInfo.ldList.size();
		progress = (float) savedSolvedConstellations / startConstCount;
		
		ldList = new ArrayList<Integer>();
		rdList = new ArrayList<Integer>();
		colList = new ArrayList<Integer>();
		startjklList = new ArrayList<Integer>();
		symList = new ArrayList<Integer>();
		for(int i = 0; i < resInfo.ldList.size(); i++) {
			ldList.add(resInfo.ldList.get(i));
			rdList.add(resInfo.rdList.get(i));
			colList.add(resInfo.colList.get(i));
			startjklList.add(resInfo.startjklList.get(i));
			symList.add(resInfo.symList.get(i));
		}
		restored = true;
	}

	@Override
	public boolean isRestored() {
		return restored;
	}
	
	@Override
	public void reset() {
		globalWorkSize = 0;
		startConstCount = 0;
		progress = 0;
		savedSolvedConstellations = 0;
		solutions = 0;
		savedSolutions = 0;
		duration = 0;
		savedDuration = 0;
		start = 0;
		end = 0;
		gpuDone = false;
		restored = false;
		System.gc();
	}

	@Override
	public long getDuration() {
		if(restored && isIdle())
			duration = savedDuration;
		else if(start != 0 && end == 0)
			duration = savedDuration + System.currentTimeMillis() - start;
		else if(end != 0)
			duration = savedDuration + end - start;
		return duration;
	}

	@Override
	public float getProgress() {
		if(startConstCount == 0)
			return 0;
		if(gpuDone || (restored && isIdle()) || start == 0)
			return progress;
		if(getDuration() == 0 || progressMem == null)
			return ((float) savedSolvedConstellations) / startConstCount;		// either has a value, is still 0 or is 0 because of reset
		
		int solvedConstellations = savedSolvedConstellations;
		synchronized(progressMem) {
			CL10.clEnqueueReadBuffer(memqueue, progressMem, CL10.CL_TRUE, 0, progressBuf, null, null);
			for(int i = 0; i < globalWorkSize; i++) {
				if(progressBuf.get(i) == 1) {
					solvedConstellations++;
				}
			}
			progress = ((float) solvedConstellations) / startConstCount;
		}
		
		return progress;
	}

	@Override
	public long getSolutions() {
		if(gpuDone)
			return solutions;
		if(resMem == null || start == 0)
			return savedSolutions;		// either has a value, is still 0 or is 0 because of reset
		
		long solutions = 0;
		synchronized(resMem) {
			CL10.clEnqueueReadBuffer(memqueue, resMem, CL10.CL_TRUE, 0, resBuf, null, null);
			for(int i = 0; i < globalWorkSize; i++) {
				solutions += resBuf.get(i) * symList.get(i);
			}
			this.solutions = solutions;
		}
		return savedSolutions + solutions;
	}

	// own functions
	private void init() throws LWJGLException {
		// intbuffer for containing error information if needed
		errBuf = BufferUtils.createIntBuffer(1);

		context = CLContext.create(platform, platform.getDevices(CL10.CL_DEVICE_TYPE_ALL), errBuf);
		
		xqueue = CL10.clCreateCommandQueue(context, device, CL10.CL_QUEUE_PROFILING_ENABLE, errBuf);
		Util.checkCLError(errBuf.get(0));
		memqueue = CL10.clCreateCommandQueue(context, device, CL10.CL_QUEUE_PROFILING_ENABLE, errBuf);
		Util.checkCLError(errBuf.get(0));

		program = CL10.clCreateProgramWithSource(context, getKernelSourceAsString("de/nqueensfaf/res/kernels.c"), null);
		String options = "-D N="+N + " -D BLOCK_SIZE="+WORKGROUP_SIZE + " -cl-mad-enable";
		int error = CL10.clBuildProgram(program, device, options, null);
		Util.checkCLError(error);

		kernel = CL10.clCreateKernel(program, "run", null);
	}

	private void transferDataToDevice() {
		if(savedDuration == 0) {		// if duration is 0, then restore() was not called
			generator = new GpuConstellationsGenerator();
			generator.genConstellations(N);
			
			ldList = generator.ldList;
			rdList = generator.rdList;
			colList = generator.colList;
			startjklList = generator.startjklList;
			symList = generator.symList;

			startConstCount = generator.startConstCount;
		}
		sortIntoWorkgroups();

		// OpenCL-Memory Objects to be passed to the kernel
		// ld
		ldMem = CL10.clCreateBuffer(context, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		ByteBuffer paramPtr = CL10.clEnqueueMapBuffer(memqueue, ldMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			paramPtr.putInt(i*4, ldList.get(i));
		}
		CL10.clEnqueueUnmapMemObject(memqueue, ldMem, paramPtr, null, null);
		
		// rd
		rdMem = CL10.clCreateBuffer(context, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		paramPtr = CL10.clEnqueueMapBuffer(memqueue, rdMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			paramPtr.putInt(i*4, rdList.get(i));
		}
		CL10.clEnqueueUnmapMemObject(memqueue, rdMem, paramPtr, null, null);
		
		// col
		colMem = CL10.clCreateBuffer(context, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		paramPtr = CL10.clEnqueueMapBuffer(memqueue, colMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			paramPtr.putInt(i*4, colList.get(i));
		}
		CL10.clEnqueueUnmapMemObject(memqueue, colMem, paramPtr, null, null);
		
		// startjkl
		startjklMem = CL10.clCreateBuffer(context, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		paramPtr = CL10.clEnqueueMapBuffer(memqueue, startjklMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			paramPtr.putInt(i*4, startjklList.get(i));
		}
		CL10.clEnqueueUnmapMemObject(memqueue, startjklMem, paramPtr, null, null);

		// result memory
		resMem = CL10.clCreateBuffer(context, CL10.CL_MEM_READ_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		synchronized(resMem) {
			Util.checkCLError(errBuf.get(0));
			ByteBuffer resWritePtr = CL10.clEnqueueMapBuffer(memqueue, resMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
			Util.checkCLError(errBuf.get(0));
			for(int i = 0; i < globalWorkSize; i++) {
				resWritePtr.putInt(i*4, 0);
			}
			CL10.clEnqueueUnmapMemObject(memqueue, resMem, resWritePtr, null, null);
			resBuf = BufferUtils.createIntBuffer(globalWorkSize);
		}

		// progress indicator memory
		progressMem = CL10.clCreateBuffer(context, CL10.CL_MEM_READ_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		synchronized(progressMem) {
			Util.checkCLError(errBuf.get(0));
			ByteBuffer progressWritePtr = CL10.clEnqueueMapBuffer(memqueue, progressMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
			Util.checkCLError(errBuf.get(0));
			for(int i = 0; i < globalWorkSize; i++) {
				progressWritePtr.putInt(i*4, 0);
			}
			CL10.clEnqueueUnmapMemObject(memqueue, progressMem, progressWritePtr, null, null);
			progressBuf = BufferUtils.createIntBuffer(globalWorkSize);
		}
		
		CL10.clFlush(memqueue);
	}

	private void sortIntoWorkgroups() {
		// binding all properties together
		record BoardProperties(int ld, int rd, int col, int startjkl, int sym) {
			BoardProperties(int ld, int rd, int col, int startjkl, int sym) {
				this.ld = ld;
				this.rd = rd;
				this.col = col;
				this.startjkl = startjkl;
				this.sym = sym;
			}
		}
		ArrayList<BoardProperties> bpList = new ArrayList<BoardProperties>();
		for(int i = 0; i < startConstCount-savedSolvedConstellations; i++) {
			bpList.add(new BoardProperties(ldList.get(i), rdList.get(i), colList.get(i), startjklList.get(i), symList.get(i)));
		}
		
		// sort by startjkl, putting constellations with equal startjkl's into the same workgroup
		var sortingLists = new ArrayList<ArrayList<BoardProperties>>();
		while(bpList.size() > 0) {
			var sameStartjklList = new ArrayList<BoardProperties>();
			int startjklWant = bpList.get(0).startjkl;
			// search in startjklList for all values equal to startjklWant
			// and add them to the current 'sameStartjklList'
			outer: while(true) {
				for(int x = 0; x < bpList.size(); x++) {
					if(bpList.get(x).startjkl == startjklWant) {
						sameStartjklList.add(bpList.get(x));
						bpList.remove(x);
						continue outer;
					}
				}
				break;
			}
			sortingLists.add(sameStartjklList);
		}
		
		var remainingConstellations = new ArrayList<BoardProperties>();
		int size, remaining;
		// now put the lists all together in bpList, thereby filling groups to fit the workgroup size
		for(var list : sortingLists) {
			size = list.size();
			remaining = size % WORKGROUP_SIZE;
			if(remaining >= WORKGROUP_FILL_MIN) {
				for(int i = 0; i < WORKGROUP_SIZE - remaining; i++) {
					list.add(new BoardProperties((1 << N) - 1, (1 << N) - 1, (1 << N) - 1, 69 << 15, 0));
				}
			} else if (remaining > 0) {
				while(list.size() % WORKGROUP_SIZE > 0) {
					remainingConstellations.add(list.get(list.size() - 1));
					list.remove(list.size() - 1);
				}
//				for(int i = 0; i < WORKGROUP_SIZE - remaining; i++) {
//					list.add(new BoardProperties((1 << N) - 1, (1 << N) - 1, (1 << N) - 1, 69 << 15, 0));
//				}
			}
			bpList.addAll(list);
		}
		System.out.println("remainingConstellations.size(): " + remainingConstellations.size());
		// add the remaining mixed constellations
		bpList.addAll(remainingConstellations);
		size = bpList.size();
		remaining = size % WORKGROUP_SIZE;
		if(remaining > 0) {
			for(int i = 0; i < WORKGROUP_SIZE - remaining; i++) {
				bpList.add(new BoardProperties((1 << N) - 1, (1 << N) - 1, (1 << N) - 1, 69 << 15, 0));
			}
		}
//		for(BoardProperties bp : bpList) {
//			if(bp.startjkl == 69 << 15) {
//				System.out.println("ld: " + bp.ld + ", rd: " + bp.rd + ", col: " + bp.col + ", sym: " + bp.sym + "-------- filler");
//			} else {
//				System.out.println("ld: " + bp.ld + ", rd: " + bp.rd + ", col: " + bp.col + ", sym: " + bp.sym);
//			}
//		}
//		System.exit(0);
		
		ldList.clear();
		rdList.clear();
		colList.clear();
		startjklList.clear();
		symList.clear();
		for(var bp : bpList) {
			ldList.add(bp.ld);
			rdList.add(bp.rd);
			colList.add(bp.col);
			startjklList.add(bp.startjkl);
			symList.add(bp.sym);
		}
		globalWorkSize = ldList.size();
	}
	
	private void explosionBoost9000() {
		// set kernel parameters
		kernel.setArg(0, ldMem);
		kernel.setArg(1, rdMem);
		kernel.setArg(2, colMem);
		kernel.setArg(3, startjklMem);
		kernel.setArg(4, resMem);
		kernel.setArg(5, progressMem);

		// create buffer of pointers defining the multi-dimensional size of the number of work units to execute
		final int dimensions = 1;
		PointerBuffer globalWorkers = BufferUtils.createPointerBuffer(dimensions);
		globalWorkers.put(0, globalWorkSize);
		PointerBuffer localWorkSize = BufferUtils.createPointerBuffer(dimensions);
		localWorkSize.put(0, WORKGROUP_SIZE);

		// wait for all memory operations to finish
		CL10.clFinish(memqueue);
		
		// run kernel and profile time
		final PointerBuffer xEventBuf = BufferUtils.createPointerBuffer(1);		// buffer for event that is used for measuring the execution time
		CL10.clEnqueueNDRangeKernel(xqueue, kernel, dimensions, null, globalWorkers, localWorkSize, null, xEventBuf);
		CL10.clFlush(xqueue);
		
		// set pseudo starttime
		start = System.currentTimeMillis();
		
		// wait for the gpu computation to finish
		CL10.clFinish(xqueue);

		// get exact time values using CLEvent
		final CLEvent event = xqueue.getCLEvent(xEventBuf.get(0));
		start = event.getProfilingInfoLong(CL10.CL_PROFILING_COMMAND_START) / 1_000_000;
		end = event.getProfilingInfoLong(CL10.CL_PROFILING_COMMAND_END) / 1_000_000;
		
		// indicator for getProgress() and getSolutions() to not read from gpu memory any more, but just return the variable
		gpuDone = true;
	}
	
	private void readResults() {
		// read result and progress memory buffers
		synchronized(resMem) {
			synchronized(progressMem) {
				CL10.clEnqueueReadBuffer(memqueue, resMem, CL10.CL_TRUE, 0, resBuf, null, null);
				CL10.clEnqueueReadBuffer(memqueue, progressMem, CL10.CL_TRUE, 0, progressBuf, null, null);
				solutions = savedSolutions;
				int solvedConstellations = savedSolvedConstellations;
				for(int i = 0; i < globalWorkSize; i++) {
					if(progressBuf.get(i) == 1) {
						solutions += resBuf.get(i) *  symList.get(i);
						solvedConstellations++;
					}
				}
				progress = ((float) solvedConstellations) / startConstCount;
			}
		}
	}
	
	private void terminate() {
		// release all CL-objects
		CL10.clReleaseKernel(kernel);
		CL10.clReleaseProgram(program);
		CL10.clReleaseMemObject(ldMem);
		CL10.clReleaseMemObject(rdMem);
		CL10.clReleaseMemObject(colMem);
		CL10.clReleaseMemObject(startjklMem);
		CL10.clReleaseMemObject(resMem);
		CL10.clReleaseMemObject(progressMem);
		CL10.clReleaseCommandQueue(xqueue);
		CL10.clReleaseContext(context);
	}
	
	// detect operating system and use corresponding native library file if available
	private static String getOS() {
		String os = System.getProperty("os.name").toLowerCase();
		if(os.contains("win")) {
			// windows
			return "win";
		} else if(os.contains("mac")) {
			// mac
			return "mac";
		} else if(os.contains("nix") || os.contains("nux") || os.contains("aix")) {
			// unix (linux etc)
			return "unix";
		} else if(os.contains("sunos")) {
			// solaris
			return "solaris";
		} else {
			// unknown os
			return os;
		}
	}
	
	private static int checkOpenCL() {
		Process clinfo;
		try {
			clinfo = Runtime.getRuntime().exec("clinfo");
			BufferedReader in = new BufferedReader(new InputStreamReader(clinfo.getInputStream()));
			String line;
			if((line = in.readLine()) != null) {
				if(line.contains(" 0") || line.contains("no usable platforms")) {
					return 0;
				} else {
					return 1;
				}
			}
		} catch (IOException e) {
			// clinfo is not installed.
			switch(getOS()) {
			case "win":
				// Good that we have the windows version in our archive!
				File clinfoFile = unpackClinfo();
				if(clinfoFile == null) {
					return -1;
				}
				try {
					clinfo = Runtime.getRuntime().exec(clinfoFile.getAbsolutePath());
					BufferedReader in = new BufferedReader(new InputStreamReader(clinfo.getInputStream()));
					String line;
					if((line = in.readLine()) != null) {
						if(line.contains(" 0") || line.contains("no usable platforms")) {
							return 0;
						} else {
							return 1;
						}
					}
				} catch (IOException e1) {
					return -1;
				}
				break;
			case "mac", "unix", "solaris":
				break;
			default:
				break;
			}
		}
		return -1;
	}
	
	private static File unpackClinfo() {
		// create temporary directory to store the clinfo file inside
		try {
			tempDir = Files.createTempDirectory("NQueensFaf");
			// copy the clinfo file from within the jar to the temporary directory
			InputStream in = GpuSolver.class.getClassLoader().getResourceAsStream("de/nqueensfaf/res/clinfo/clinfo.exe");
			byte[] buffer = new byte[1024];
			int read = -1;
			File file = new File(tempDir + "/clinfo.exe");
			FileOutputStream fos = new FileOutputStream(file);
			while((read = in.read(buffer)) != -1) {
				fos.write(buffer, 0, read);
			}
			fos.close();
			in.close();
			return file;
		} catch (IOException e1) {
			e1.printStackTrace();
			return null;
		}
	}
	
	private static void loadLWJGLNative() {
		String filenameIn = null;
		String filenameOut = null;

		// determine system architecture
		String arch = System.getProperty("os.arch");
		if(arch.contains("64"))
			arch = "64";
		else
			arch = "";
		// determine operating system
		switch(getOS()) {
		case "win":
			filenameIn = "lwjgl" + arch + ".dll";
			filenameOut = filenameIn;
			break;
		case "mac":
			filenameIn = "liblwjgl_mac.dylib";
			filenameOut = "liblwjgl.dylib";
			break;
		case "unix":
			filenameIn = "liblwjgl" + arch + "_linux.so";
			filenameOut = "liblwjgl" + arch + ".so";
			break;
		case "solaris":
			filenameIn = "liblwjgl" + arch + "_solaris.so";
			filenameOut = "liblwjgl" + arch + ".so";
			break;
		default:
			System.err.println("No native executables available for this operating system (" + getOS() + ").");
			return;
		}
		try {
			// create temporary directory to store the lwjgl binary file inside, if it does not exist yet
			if(tempDir == null)
				tempDir = Files.createTempDirectory("NQueensFaf");
			// copy the lwjgl binary file from within the jar to the temporary directory
			InputStream in = GpuSolver.class.getClassLoader().getResourceAsStream("de/nqueensfaf/res/lwjgl/" + filenameIn);
			byte[] buffer = new byte[1024];
			int read = -1;
			File file = new File(tempDir + "/" + filenameOut);
			FileOutputStream fos = new FileOutputStream(file);
			while((read = in.read(buffer)) != -1) {
				fos.write(buffer, 0, read);
			}
			fos.close();
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.setProperty("org.lwjgl.librarypath", tempDir.toAbsolutePath().toString());
	}

	private static void deleteTempDir() {
		try {
			// platform specific binary
			String prefix = "";
			String suffix = "";
			switch(getOS()) {
			case "win":
				prefix = "wscript.exe ";
				suffix = ".vbs";
				break;
			case "mac":
			case "unix":
			case "solaris":
				prefix = "sh ";
				suffix = ".sh";
				break;
			default:
				System.err.println("No cleanup-executable available for this operating system (" + getOS() + ").");
				return;
			}

			// if there is a binary for this operating system, use it to clean up the temporary files created by this program ( -> lwjgl-binaries)
			if(suffix.length() > 0) {
				// clean the temp-directory that was created for the lwjgl-native binary
				InputStream in = GpuSolver.class.getClassLoader().getResourceAsStream("de/nqueensfaf/res/NQueensFaf_Cleanup" + suffix);
				byte[] buffer = new byte[1024];
				int read = -1;
				File file = File.createTempFile("NQueensFaf_Cleanup", suffix);
				FileOutputStream fos = new FileOutputStream(file);
				while((read = in.read(buffer)) != -1) {
					fos.write(buffer, 0, read);
				}
				fos.close();
				in.close();

				Runtime.getRuntime().exec(prefix + file.getAbsolutePath());
			}
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}
	
	public String[] getAvailableDevices() {
		if(!openclable) {
			throw new IllegalStateException("No OpenCL-capable device was found. GpuSolver is not available.");
		}
		if(devices == null) {
			devices = new ArrayList<CLDevice>();
		} else {
			devices.clear();
		}
		ArrayList<String> deviceNames = new ArrayList<String>();
		try {
			for(CLPlatform platform : CLPlatform.getPlatforms()) {
				for(CLDevice device : platform.getDevices(CL10.CL_DEVICE_TYPE_GPU)) {
					devices.add(device);
					deviceNames.add(device.getInfoString(CL10.CL_DEVICE_NAME));
				}
			}
		} catch(NullPointerException e) {
			throw e;
		}
		String[] arr = new String[deviceNames.size()];
		for(int i = 0; i < arr.length; i++) {
			arr[i] = deviceNames.get(i);
		}
		return arr;
	}

	public void setDevice(int idx) {
		if(!openclable) {
			throw new IllegalStateException("No OpenCL-capable device was found. GpuSolver is not available.");
		}
		if(idx < 0 || idx >= devices.size()) {
			throw new IllegalArgumentException("Invalid index value: " + idx + " (size:" + devices.size() + ")");
		}
		device = devices.get(idx);
		platform = device.getPlatform();
	}

	private String getKernelSourceAsString(String filepath) {
		BufferedReader br = null;
		String resultString = null;
		try {
			InputStream clSourceFile = GpuSolver.class.getClassLoader().getResourceAsStream(filepath);
			br = new BufferedReader(new InputStreamReader(clSourceFile));
			String line = null;
			StringBuilder result = new StringBuilder();
			while((line = br.readLine()) != null) {
				result.append(line);
				result.append("\n");
			}
			resultString = result.toString();
		} catch(NullPointerException e) {
			e.printStackTrace();
		} catch(IOException e) {
			e.printStackTrace();
		} finally {
			try {
				br.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
		return resultString;
	}

	// getters and setters
	
	// returns the total number of enqueued work-items 
	public int getGlobalWorkSize() {
		return globalWorkSize;
	}
	
	// gets WORKGROUP_SIZE
	public int getWorkgroupSize() {
		return WORKGROUP_SIZE;
	}
	
	// sets WORKGROUP_SIZE
	public void setWorkgroupSize(int s) {
		if(device == null) {
			throw new IllegalStateException("Choose a device first");
		}
		int maxWorkgroupSize = device.getInfoInt(CL10.CL_DEVICE_MAX_WORK_GROUP_SIZE);
		if(s <= 0 || s >= maxWorkgroupSize) {
			throw new IllegalArgumentException("WorkgroupSize must be between 0 and " + maxWorkgroupSize + " (=max for this device)");
		}
		WORKGROUP_SIZE = s;
		WORKGROUP_FILL_MIN = (int) ((1-0.15625) * WORKGROUP_SIZE) + 5;
	}
	
	// record class for saving and restoring
	private record RestorationInformation(int N, long duration, long solutions, int startConstCount, 
			ArrayList<Integer> ldList, ArrayList<Integer> rdList, ArrayList<Integer> colList, ArrayList<Integer> startjklList, ArrayList<Integer> symList) implements Serializable {
		RestorationInformation(int N, long duration, long solutions, int startConstCount, 
				ArrayList<Integer> ldList, ArrayList<Integer> rdList, ArrayList<Integer> colList, ArrayList<Integer> startjklList, ArrayList<Integer> symList) {
			this.N = N;
			this.duration = duration;
			this.solutions = solutions;
			this.startConstCount = startConstCount;
			this.ldList = ldList;
			this.rdList = rdList;
			this.colList = colList;
			this.startjklList = startjklList;
			this.symList = symList;
		}
	}
}
