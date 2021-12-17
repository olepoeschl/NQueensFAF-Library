package de.nqueensfaf;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import de.nqueensfaf.compute.CpuSolver;
import de.nqueensfaf.compute.GpuSolver;

class SolverTest {

	Solver solver;
	
	@BeforeAll
	static void setUpBeforeClass() throws Exception {
	}

	@AfterAll
	static void tearDownAfterClass() throws Exception {
	}

	@BeforeEach
	void setUp() throws Exception {
	}

	@AfterEach
	void tearDown() throws Exception {
	}

	// Manual test
//	@Test
//	void testCpuSolver() {
//		CpuSolver s = new CpuSolver();
//		s.setN(6);
//		s.addTerminationCallback(() -> {
//			System.out.println(s.getSolutions() + " solutions found in " + s.getDuration() + " ms");
//		});
//		s.solve();
//	}

	// Manual test
	@Test
	void testGpuSolver() {
//		try {
//			// for making sure the profiler (VisualVM) is ready
//			Thread.sleep(5000);
//		} catch (InterruptedException e1) {
//			e1.printStackTrace();
//		}
		GpuSolver s = new GpuSolver();
		s.setDevice(0);
		s.setProgressUpdatesEnabled(false);
		s.setN(17);
		new Thread(() -> {
			while(true) {
				if(s.getGlobalWorkSize() == 0) {
					try {
						Thread.sleep(100);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
					continue;
				}
				System.out.println("globalWorkSize: " + s.getGlobalWorkSize());
				break;
			}
		}).start();
		s.setOnProgressUpdateCallback((progress, solutions) -> {
			System.out.println("progress: " + progress + ", solutions: " + solutions);
		});
		s.addTerminationCallback(() -> {
			System.out.println(s.getSolutions() + " solutions found in " + s.getDuration() + " ms");
		});
		s.solve();
	}
	
	/*
	// Manual test
	@Test
	void test() {
		solver = new MockSolver();
		solver.setN(5);
		solver.addInitializationCallback(() -> System.out.println("starte..."));
		solver.addTerminationCallback(() -> System.out.println("fertig!"));
		solver.setOnTimeUpdateCallback((duration) -> {
			System.out.println("Duration: " + duration/1000 + "." + duration%1000);
		});
		solver.setTimeUpdateDelay(500);
		solver.setOnProgressUpdateCallback((progress, solutions) -> {
			System.out.println("Progress: " + progress + ", solutions: " + solutions);
		});
		solver.setProgressUpdateDelay(1000);
		solver.solve();
		
		solver.reset();
		solver.solveAsync();
		while(solver.isIdle() || solver.isInitializing());
		while(solver.isRunning()) {
			System.out.println("läuft");
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			if(solver.getDuration() > 2500)
				break;
		}
		System.out.println("warte auf beendigung...");
		try {
			solver.waitFor();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	@Test
	void testFail() {
		// setN
		assertThrows(IllegalArgumentException.class, () -> new MockSolver().setN(-2));
		assertThrows(IllegalArgumentException.class, () -> new MockSolver().setN(32));
		solver = new MockSolver();
		solver.setN(2);
		solver.solveAsync();
		assertThrows(IllegalStateException.class, () -> solver.setN(15));
		try {
			solver.waitFor();
		} catch (InterruptedException e1) {
			e1.printStackTrace();
		}
		
		// set<X>UpdateDelay
		assertThrows(IllegalArgumentException.class, () -> solver.setTimeUpdateDelay(-7));
		assertThrows(IllegalArgumentException.class, () -> solver.setProgressUpdateDelay(-13));
		
		// solve, solveAsync
		solver = new MockSolver();
		assertThrows(IllegalStateException.class, () -> solver.solve());
		solver.setN(1);
		solver.solveAsync();
		assertThrows(IllegalStateException.class, () -> solver.solve());
		assertThrows(IllegalStateException.class, () -> solver.solveAsync());
		try {
			solver.waitFor();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		// initialization, termination
		assertThrows(IllegalArgumentException.class, () -> solver.addInitializationCallback(null));
		assertThrows(IllegalArgumentException.class, () -> solver.addTerminationCallback(null));
	}
	
	class MockSolver extends Solver {
		int total, done;
		long start, end;
		
		@Override
		protected void run() {
			total = 100*N;
			start = System.currentTimeMillis();
			for(done = 0; done < total; done++) {
				end = System.currentTimeMillis();
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}

		@Override
		public void store(String filename) {}
		@Override
		public void restore(String filename) {}

		@Override
		public void reset() {
			done = 0;
			start = 0;
			end = 0;
		}
		
		@Override
		public long getDuration() {
			return end - start;
		}

		@Override
		public float getProgress() {
			return ((float) done) / total;
		}

		@Override
		public long getSolutions() {
			return done*10;
		}

		@Override
		public boolean isRestored() {
			return false;
		}
	}
	*/
}
