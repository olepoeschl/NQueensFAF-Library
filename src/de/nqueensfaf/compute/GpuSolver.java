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

import de.nqueensfaf.Solver;

public class GpuSolver extends Solver {

	static {
		// enables the easy use of lwjgl out of the jar-archive. The customer only need the lwjgl.jar. (LWJGL 2.9.3)
		prepareLWJGLNative();
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
	private CLMem ldMem, rdMem, colMem, LDMem, RDMem, klMem, startMem, resMem, progressMem;
	private int computeUnits;
	private final int WORKGROUP_SIZE = 64;
	private int globalWorkSize;

	// calculation related stuff
	private GpuConstellationsGenerator generator;
	private int startConstCount;
	private int[] ldArr, rdArr, colArr, LDArr, RDArr, klArr, startArr, symArr;
	private long solutions, savedSolutions;
	private long duration, start, end, savedDuration;
	private float progress;
	private int savedSolvedConstellations;
	
	// control flow variables
	private boolean gpuDone = false;
	
	// inherited functions
	@Override
	protected void run() {
		if(start != 0) {
			throw new IllegalStateException("You first have to call reset() when calling solve() multiple times on the same object");
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
	}

	@Override
	public void save(String filepath) throws IOException {
		// if Solver was not even started yet, throw exception
		if(start == 0) {
			throw new IllegalStateException("Nothing to be saved");
		}
		long solutions = savedSolutions;
		ArrayList<Integer> 
				ldList = new ArrayList<Integer>(),
				rdList = new ArrayList<Integer>(),
				colList = new ArrayList<Integer>(),
				LDList = new ArrayList<Integer>(),
				RDList = new ArrayList<Integer>(),
				klList = new ArrayList<Integer>(),
				startList = new ArrayList<Integer>(),
				symList = new ArrayList<Integer>();
		synchronized(resMem) {
			synchronized(progressMem) {
				CL10.clEnqueueReadBuffer(memqueue, resMem, CL10.CL_TRUE, 0, resBuf, null, null);
				CL10.clEnqueueReadBuffer(memqueue, progressMem, CL10.CL_TRUE, 0, progressBuf, null, null);
				for(int i = 0; i < startConstCount; i++) {
					solutions += resBuf.get(i) * this.symArr[i];
					if(progressBuf.get(i) == 1) {
						ldList.add(ldArr[i]);
						rdList.add(rdArr[i]);
						colList.add(colArr[i]);
						LDList.add(LDArr[i]);
						RDList.add(RDArr[i]);
						klList.add(klArr[i]);
						startList.add(startArr[i]);
						symList.add(symArr[i]);
					}
				}
			}
		}
		RestorationInformation resInfo = new RestorationInformation(getDuration(), solutions, startConstCount, ldList, rdList, colList, LDList, RDList, klList, startList, symList);

		FileOutputStream fos = new FileOutputStream(filepath);
		ObjectOutputStream oos = new ObjectOutputStream(fos);
		oos.writeObject(resInfo);
		oos.flush();
		oos.close();
		fos.close();
	}

	@Override
	public void restore(String filepath) throws IOException, ClassNotFoundException {
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
		savedDuration = resInfo.duration;
		savedSolutions = resInfo.solutions;
		startConstCount = resInfo.startConstCount;
		savedSolvedConstellations = startConstCount - resInfo.ldList.size();
		
		// fill the constellation arrays
		globalWorkSize = startConstCount;
		// if needed, round globalWorkSize up to the next matching number
		computeUnits = device.getInfoInt(CL10.CL_DEVICE_MAX_COMPUTE_UNITS);
		if(globalWorkSize % (WORKGROUP_SIZE * computeUnits) != 0) {
			globalWorkSize = startConstCount - (startConstCount % (WORKGROUP_SIZE * computeUnits)) + (WORKGROUP_SIZE * computeUnits);
		}
		ldArr = new int[globalWorkSize];
		rdArr = new int[globalWorkSize];
		colArr = new int[globalWorkSize];
		LDArr = new int[globalWorkSize];
		RDArr = new int[globalWorkSize];
		klArr = new int[globalWorkSize];
		startArr = new int[globalWorkSize];
		symArr = new int[globalWorkSize];
		for(int i = 0; i < startConstCount; i++) {
			ldArr[i] = resInfo.ldList.get(i);
			rdArr[i] = resInfo.rdList.get(i);
			colArr[i] = resInfo.colList.get(i);
			LDArr[i] = resInfo.LDList.get(i);
			RDArr[i] = resInfo.RDList.get(i);
			klArr[i] = resInfo.klList.get(i);
			startArr[i] = resInfo.startList.get(i);
			symArr[i] = resInfo.symList.get(i);
		}
	}

	@Override
	public void reset() {
		progress = 0;
		savedSolvedConstellations = 0;
		solutions = 0;
		savedSolutions = 0;
		duration = 0;
		savedDuration = 0;
		start = 0;
		end = 0;
		gpuDone = false;
		System.gc();
	}

	@Override
	public long getDuration() {
		if(start != 0 && end == 0)
			duration = savedDuration + System.currentTimeMillis() - start;
		else if(end != 0)
			duration = savedDuration + end - start;
		return duration;
	}

	@Override
	public float getProgress() {
		if(gpuDone) {
			return progress;
		}
		int solvedConstellations = savedSolvedConstellations;
		// calculate current solvecounter
		synchronized(progressMem) {
			CL10.clEnqueueReadBuffer(memqueue, progressMem, CL10.CL_TRUE, 0, progressBuf, null, null);
			for(int i = 0; i < startConstCount; i++) {
				solvedConstellations += progressBuf.get(i);
			}
			progress = ((float) solvedConstellations) / startConstCount;
		}
		return progress;
	}

	@Override
	public long getSolutions() {
		if(gpuDone) {
			return solutions;
		}
		long solutions = 0;
		synchronized(resMem) {
			CL10.clEnqueueReadBuffer(memqueue, resMem, CL10.CL_TRUE, 0, resBuf, null, null);
			for(int i = 0; i < startConstCount; i++) {
				solutions += resBuf.get(i) * symArr[i];
			}
			this.solutions = solutions;
		}
		return savedSolutions + solutions;
	}

	// own functions
	private void init() throws LWJGLException {
		// load OpenCL native library
		CL.create();
		
		// intbuffer for containing error information if needed
		errBuf = BufferUtils.createIntBuffer(1);

		try {
			context = CLContext.create(platform, platform.getDevices(CL10.CL_DEVICE_TYPE_ALL), errBuf);
		} catch (LWJGLException e) {
			e.printStackTrace();
		}
		xqueue = CL10.clCreateCommandQueue(context, device, CL10.CL_QUEUE_PROFILING_ENABLE, errBuf);
		Util.checkCLError(errBuf.get(0));
		memqueue = CL10.clCreateCommandQueue(context, device, CL10.CL_QUEUE_PROFILING_ENABLE, errBuf);
		Util.checkCLError(errBuf.get(0));

		program = CL10.clCreateProgramWithSource(context, getKernelSourceAsString("res/kernels.c"), null);
		String options = "-D N="+getN() + " -D BLOCK_SIZE="+WORKGROUP_SIZE + " -cl-mad-enable";
		int error = CL10.clBuildProgram(program, device, options, null);
		Util.checkCLError(error);

		kernel = CL10.clCreateKernel(program, "run", null);
	}

	private void transferDataToDevice() {
		if(savedDuration == 0) {		// if duration is 0, then restore() was not called
			generator = new GpuConstellationsGenerator();
			generator.genConstellations(N);
			startConstCount = generator.getStartConstCount();
			
			globalWorkSize = startConstCount;
			// if needed, round globalWorkSize up to the next matching number
			computeUnits = device.getInfoInt(CL10.CL_DEVICE_MAX_COMPUTE_UNITS);
			if(globalWorkSize % (WORKGROUP_SIZE * computeUnits) != 0) {
				globalWorkSize = startConstCount - (startConstCount % (WORKGROUP_SIZE * computeUnits)) + (WORKGROUP_SIZE * computeUnits);
			}
			
			ldArr = new int[globalWorkSize];
			rdArr = new int[globalWorkSize];
			colArr = new int[globalWorkSize];
			LDArr = new int[globalWorkSize];
			RDArr = new int[globalWorkSize];
			klArr = new int[globalWorkSize];
			startArr = new int[globalWorkSize];
			symArr = new int[globalWorkSize];
			for(int i = 0; i < startConstCount; i++) {
				ldArr[i] = generator.ldList.removeFirst();
				rdArr[i] = generator.rdList.removeFirst();
				colArr[i] = generator.colList.removeFirst();
				LDArr[i] = generator.LDList.removeFirst();
				RDArr[i] = generator.RDList.removeFirst();
				klArr[i] = generator.klList.removeFirst();
				startArr[i] = generator.startList.removeFirst();
				symArr[i] = generator.symList.removeFirst();
			}
		}
		// fill the newly created task slots in globalWorkSize using empty tasks (-> kernels.c)
		for(int i = startConstCount; i < globalWorkSize; i++) {
			ldArr[i] = 0xFFFFFFFF;
			rdArr[i] = 0xFFFFFFFF;
			colArr[i] = 0xFFFFFFFF;
			LDArr[i] = 0xFFFFFFFF;
			RDArr[i] = 0xFFFFFFFF;
			klArr[i] = 0xFFFFFFFF;
			startArr[i] = 0xFFFFFFFF;
			symArr[i] = 0xFFFFFFFF;
		}

		// OpenCL-Memory Objects to be passed to the kernel
		// ld
		ldMem = CL10.clCreateBuffer(context, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		ByteBuffer paramPtr = CL10.clEnqueueMapBuffer(memqueue, ldMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			paramPtr.putInt(i*4, ldArr[i]);
		}
		CL10.clEnqueueUnmapMemObject(memqueue, ldMem, paramPtr, null, null);
		
		// rd
		rdMem = CL10.clCreateBuffer(context, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		paramPtr = CL10.clEnqueueMapBuffer(memqueue, rdMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			paramPtr.putInt(i*4, rdArr[i]);
		}
		CL10.clEnqueueUnmapMemObject(memqueue, rdMem, paramPtr, null, null);
		
		// col
		colMem = CL10.clCreateBuffer(context, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		paramPtr = CL10.clEnqueueMapBuffer(memqueue, colMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			paramPtr.putInt(i*4, colArr[i]);
		}
		CL10.clEnqueueUnmapMemObject(memqueue, colMem, paramPtr, null, null);
		
		// LD
		LDMem = CL10.clCreateBuffer(context, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		paramPtr = CL10.clEnqueueMapBuffer(memqueue, LDMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			paramPtr.putInt(i*4, LDArr[i]);
		}
		CL10.clEnqueueUnmapMemObject(memqueue, LDMem, paramPtr, null, null);
		
		// RD
		RDMem = CL10.clCreateBuffer(context, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		paramPtr = CL10.clEnqueueMapBuffer(memqueue, RDMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			paramPtr.putInt(i*4, RDArr[i]);
		}
		CL10.clEnqueueUnmapMemObject(memqueue, RDMem, paramPtr, null, null);
		
		// kl
		klMem = CL10.clCreateBuffer(context, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		paramPtr = CL10.clEnqueueMapBuffer(memqueue, klMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			paramPtr.putInt(i*4, klArr[i]);
		}
		CL10.clEnqueueUnmapMemObject(memqueue, klMem, paramPtr, null, null);
		
		// start
		startMem = CL10.clCreateBuffer(context, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		paramPtr = CL10.clEnqueueMapBuffer(memqueue, startMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			paramPtr.putInt(i*4, startArr[i]);
		}
		CL10.clEnqueueUnmapMemObject(memqueue, startMem, paramPtr, null, null);

		// result memory
		resMem = CL10.clCreateBuffer(context, CL10.CL_MEM_READ_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		ByteBuffer resWritePtr = CL10.clEnqueueMapBuffer(memqueue, resMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			resWritePtr.putInt(i*4, 0);
		}
		CL10.clEnqueueUnmapMemObject(memqueue, resMem, resWritePtr, null, null);
		resBuf = BufferUtils.createIntBuffer(globalWorkSize);

		// progress indicator memory
		progressMem = CL10.clCreateBuffer(context, CL10.CL_MEM_READ_ONLY | CL10.CL_MEM_ALLOC_HOST_PTR, globalWorkSize*4, errBuf);
		Util.checkCLError(errBuf.get(0));
		ByteBuffer progressWritePtr = CL10.clEnqueueMapBuffer(memqueue, progressMem, CL10.CL_TRUE, CL10.CL_MAP_WRITE, 0, globalWorkSize*4, null, null, errBuf);
		Util.checkCLError(errBuf.get(0));
		for(int i = 0; i < globalWorkSize; i++) {
			progressWritePtr.putInt(i*4, 0);
		}
		CL10.clEnqueueUnmapMemObject(memqueue, progressMem, progressWritePtr, null, null);
		progressBuf = BufferUtils.createIntBuffer(globalWorkSize);
	}

	private void explosionBoost9000() {
		// set kernel parameters
		kernel.setArg(0, ldMem);
		kernel.setArg(1, rdMem);
		kernel.setArg(2, colMem);
		kernel.setArg(3, LDMem);
		kernel.setArg(4, RDMem);
		kernel.setArg(5, klMem);
		kernel.setArg(6, startMem);
		kernel.setArg(7, resMem);
		kernel.setArg(8, progressMem);

		// create buffer of pointers defining the multi-dimensional size of the number of work units to execute
		final int dimensions = 1;
		PointerBuffer globalWorkers = BufferUtils.createPointerBuffer(dimensions);
		globalWorkers.put(0, globalWorkSize);
		PointerBuffer localWorkSize = BufferUtils.createPointerBuffer(dimensions);
		localWorkSize.put(0, WORKGROUP_SIZE);

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
		start = event.getProfilingInfoLong(CL10.CL_PROFILING_COMMAND_START) / 1000000;
		end = event.getProfilingInfoLong(CL10.CL_PROFILING_COMMAND_END) / 1000000;
		
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
				for(int i = 0; i < startConstCount; i++) {
					solutions += resBuf.get(i) *  symArr[i];
					solvedConstellations += progressBuf.get(i);
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
		CL10.clReleaseMemObject(LDMem);
		CL10.clReleaseMemObject(RDMem);
		CL10.clReleaseMemObject(klMem);
		CL10.clReleaseMemObject(startMem);
		CL10.clReleaseMemObject(resMem);
		CL10.clReleaseMemObject(progressMem);
		CL10.clReleaseCommandQueue(xqueue);
		CL10.clReleaseContext(context);
		// unload OpenCL native library
		CL.destroy();
	}
	
	// detect operating system and use corresponding native library file if available
	private static void prepareLWJGLNative() {
		Path temp_libdir = null;
		String filenameIn = null;
		String filenameOut = null;

		// determine system architecture
		String arch = System.getProperty("os.arch");
		if(arch.contains("64"))
			arch = "64";
		else
			arch = "";
		// determine operating system
		String os = System.getProperty("os.name").toLowerCase();
		if(os.contains("win")) {
			// windows
			filenameIn = "lwjgl" + arch + ".dll";
			filenameOut = filenameIn;
		} else if(os.contains("mac")) {
			// mac
			filenameIn = "liblwjgl_mac.dylib";
			filenameOut = "liblwjgl.dylib";
		} else if(os.contains("nix") || os.contains("nux") || os.contains("aix")) {
			// unix (linux etc)
			filenameIn = "liblwjgl" + arch + "_linux.so";
			filenameOut = "liblwjgl" + arch + ".so";
		} else if(os.contains("sunos")) {
			// solaris
			filenameIn = "liblwjgl" + arch + "_solaris.so";
			filenameOut = "liblwjgl" + arch + ".so";
		} else {
			// unknown os
			System.err.println("No native executables available for this operating system (" + os + ").");
		}
		try {
			// create temporary directory to store the native files inside
			temp_libdir = Files.createTempDirectory("NQueensFaf");

			// copy the native file from within the jar to the temporary directory
			InputStream in = GpuSolver.class.getClassLoader().getResourceAsStream("de/nqueensfaf/res/natives/" + filenameIn);
			byte[] buffer = new byte[1024];
			int read = -1;
			File file = new File(temp_libdir + "/" + filenameOut);
			FileOutputStream fos = new FileOutputStream(file);
			while((read = in.read(buffer)) != -1) {
				fos.write(buffer, 0, read);
			}
			fos.close();
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.setProperty("org.lwjgl.librarypath", temp_libdir.toAbsolutePath().toString());
	}

	public String[] getAvailableDevices() {
		if(devices != null) {
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
		if(idx < 0 || idx >= devices.size()) {
			throw new IllegalArgumentException("Invalid index value: " + idx);
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

	// for saving and restoring
	private record RestorationInformation(long duration, long solutions, int startConstCount, 
			ArrayList<Integer> ldList, ArrayList<Integer> rdList, ArrayList<Integer> colList, ArrayList<Integer> LDList, ArrayList<Integer> RDList, 
			ArrayList<Integer> klList, ArrayList<Integer> startList, ArrayList<Integer> symList) implements Serializable {
		RestorationInformation(long duration, long solutions, int startConstCount, 
				ArrayList<Integer> ldList, ArrayList<Integer> rdList, ArrayList<Integer> colList, ArrayList<Integer> LDList, ArrayList<Integer> RDList, 
				ArrayList<Integer> klList, ArrayList<Integer> startList, ArrayList<Integer> symList) {
			this.duration = 0;
			this.solutions = 0;
			this.startConstCount = 0;
			this.ldList = null;
			this.rdList = null;
			this.colList = null;
			this.LDList = null;
			this.RDList = null;
			this.klList = null;
			this.startList = null;
			this.symList = null;
		}
	}



	// for testing
	public static void main(String[] args) {
		GpuSolver s = new GpuSolver();
		s.setN(17);
		String[] deviceNames = s.getAvailableDevices();
		for(String deviceName : deviceNames) {
			System.out.println(deviceName);
		}
	}
}
