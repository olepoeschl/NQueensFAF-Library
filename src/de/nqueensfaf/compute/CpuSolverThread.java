package de.nqueensfaf.compute;

import java.util.ArrayDeque;
import java.util.ArrayList;

class CpuSolverThread extends Thread {

	private final int N, N3, N4, N5, L, L3, L4;			// boardsize
	private long tempcounter = 0, solvecounter = 0;			// tempcounter is #(unique solutions) of current start constellation, solvecounter is #(all solutions)
	private int done = 0;						// #(done start constellations)

	private int mark1, mark2, mark3;
 
	// list of uncalculated starting positions, their indices
	private ArrayDeque<Integer> startConstellations, ldList, rdList, colList, startQueensIjklList;
	
	// for pausing or cancelling the run
	private boolean cancel = false, running = false;
	private int pause = 0;
	private CpuSolver caller;

	CpuSolverThread(CpuSolver caller, int N, ArrayDeque<Integer> startConstellations, ArrayDeque<Integer> ldList, 
			ArrayDeque<Integer> rdList, ArrayDeque<Integer> colList, ArrayDeque<Integer> startQueensIjklList) {
		this.caller = caller;
		this.N = N;
		N3 = N - 3;
		N4 = N - 4;
		N5 = N - 5;
		L = 1 << (N-1);
		L3 = 1 << N3;
		L4 = 1 << N4;
		this.startConstellations = startConstellations;
		this.ldList = ldList;
		this.rdList = rdList;
		this.colList = colList;
		this.startQueensIjklList = startQueensIjklList;
	}

	// Recursive functions for Placing the Queens

	// for N-1-j = 0
	private void SQd0B(int ld, int rd, int col, int idx, int free) {
		if(idx == N4) {
			tempcounter++;
			return;
		}

		int bit;
		int nextfree;

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			
			int next_ld = ((ld|bit)<<1);
			int next_rd = ((rd|bit)>>1);
			int next_col = (col|bit);
			nextfree = ~(next_ld | next_rd | next_col);
			if(nextfree > 0)
				if(idx < N5) {
					if(~((next_ld<<1) | (next_rd>>1) | (next_col)) > 0)
						SQd0B(next_ld, next_rd, next_col, idx+1, nextfree);
				} else {
					SQd0B(next_ld, next_rd, next_col, idx+1, nextfree);
				}
		}
	}

	private void SQd0BkB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark1) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<2) | ((rd|bit)>>2) | (col|bit) | L3);
				if(nextfree > 0)
					SQd0B((ld|bit)<<2, ((rd|bit)>>2) | L3, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd0BkB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	// for N-1-j = 1
	private void SQd1BklB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark2) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<3) | ((rd|bit)>>3) | (col|bit) | 1 | L4);
				if(nextfree > 0)
					SQd1B(((ld|bit)<<3) | 1, ((rd|bit)>>3) | L4, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd1BklB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQd1B(int ld, int rd, int col, int idx, int free) {
		if(idx == N5) {
			tempcounter++;
			return;
		}

		int bit;
		int nextfree;

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			
			int next_ld = ((ld|bit)<<1);
			int next_rd = ((rd|bit)>>1);
			int next_col = (col|bit);
			nextfree = ~(next_ld | next_rd | next_col);
			if(nextfree > 0)
				if(idx < N5-1) {
					if(~((next_ld<<1) | (next_rd>>1) | (next_col)) > 0)
						SQd1B(next_ld, next_rd, next_col, idx+1, nextfree);
				} else {
					SQd1B(next_ld, next_rd, next_col, idx+1, nextfree);
				}
		}
	}

	private void SQd1BkBlB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark1) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<2) | ((rd|bit)>>2) | (col|bit) | L3);
				if(nextfree > 0)
					SQd1BlB(((ld|bit)<<2), ((rd|bit)>>2) | L3, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd1BkBlB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQd1BlB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark2) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				
				int next_ld = ((ld|bit)<<2) | 1;
				int next_rd = ((rd|bit)>>2);
				int next_col = (col|bit);
				nextfree = ~(next_ld | next_rd | next_col);
				if(nextfree > 0)
					if(idx < N5-2) {
						if(~((next_ld<<1) | (next_rd>>1) | (next_col)) > 0)
							SQd1B(next_ld, next_rd, next_col, idx+1, nextfree);
					} else {
						SQd1B(next_ld, next_rd, next_col, idx+1, nextfree);
					}
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd1BlB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQd1BlkB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark2) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<3) | ((rd|bit)>>3) | (col|bit) | 2 | L3);
				if(nextfree > 0)
					SQd1B(((ld|bit)<<3) | 2, ((rd|bit)>>3) | L3, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd1BlkB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQd1BlBkB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark1) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<2) | ((rd|bit)>>2) | (col|bit) | 1);
				if(nextfree > 0)
					SQd1BkB(((ld|bit)<<2) | 1, (rd|bit)>>2, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd1BlBkB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQd1BkB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark2) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<2) | ((rd|bit)>>2) | (col|bit) | L3);
				if(nextfree > 0)
					SQd1B(((ld|bit)<<2), ((rd|bit)>>2) | L3, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd1BkB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	// all following SQ functions for N-1-j > 2
	private void SQBkBlBjrB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark1) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<2) | ((rd|bit)>>2) | (col|bit) | (1 << (N3)));
				if(nextfree > 0)
					SQBlBjrB(((ld|bit)<<2), ((rd|bit)>>2) | (1 << (N3)), col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQBkBlBjrB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQBlBjrB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark2) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<2) | ((rd|bit)>>2) | (col|bit) | 1);
				if(nextfree > 0)
					SQBjrB(((ld|bit)<<2) | 1, (rd|bit)>>2, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQBlBjrB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQBjrB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark3) {
			free &= (~1);
			ld |= 1;
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
				if(nextfree > 0)
					SQB(((ld|bit)<<1), (rd|bit)>>1, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQBjrB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQB(int ld, int rd, int col, int idx, int free) {
		if(idx == N5) {
			tempcounter++;
			return;
		}

		int bit;
		int nextfree;

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			
			int next_ld = ((ld|bit)<<1);
			int next_rd = ((rd|bit)>>1);
			int next_col = (col|bit);
			nextfree = ~(next_ld | next_rd | next_col);
			if(nextfree > 0)
				if(idx < N5-1) {
					if(~((next_ld<<1) | (next_rd>>1) | (next_col)) > 0)
						SQB(next_ld, next_rd, next_col, idx+1, nextfree);
				} else {
					SQB(next_ld, next_rd, next_col, idx+1, nextfree);
				}
		}
	}

	private void SQBlBkBjrB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark1) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<2) | ((rd|bit)>>2) | (col|bit) | 1);
				if(nextfree > 0)
					SQBkBjrB(((ld|bit)<<2) | 1, (rd|bit)>>2, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQBlBkBjrB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQBkBjrB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark2) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<2) | ((rd|bit)>>2) | (col|bit) | L3);
				if(nextfree > 0)
					SQBjrB(((ld|bit)<<2), ((rd|bit)>>2) | L3, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQBkBjrB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQBklBjrB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark2) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<3) | ((rd|bit)>>3) | (col|bit) | L4 | 1);
				if(nextfree > 0)
					SQBjrB(((ld|bit)<<3) | 1, ((rd|bit)>>3) | L4, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQBklBjrB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQBlkBjrB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark2) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<3) | ((rd|bit)>>3) | (col|bit) | L3 | 2);
				if(nextfree > 0)
					SQBjrB(((ld|bit)<<3) | 2, ((rd|bit)>>3) | L3, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQBlkBjrB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	// for N-1-j = 2
	private void SQd2BlkB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark2) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<3) | ((rd|bit)>>3) | (col|bit) | L3 | 2);
				if(nextfree > 0)
					SQd2B(((ld|bit)<<3) | 2, ((rd|bit)>>3) | L3, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd2BlkB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQd2BklB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark2) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<3) | ((rd|bit)>>3) | (col|bit) | L4 | 1);
				if(nextfree > 0)
					SQd2B(((ld|bit)<<3) | 1, ((rd|bit)>>3) | L4, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd2BklB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQd2BlBkB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark1) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<2) | ((rd|bit)>>2) | (col|bit) | 1);
				if(nextfree > 0)
					SQd2BkB(((ld|bit)<<2) | 1, (rd|bit)>>2, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd2BlBkB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQd2BkBlB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark1) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<2) | ((rd|bit)>>2) | (col|bit) | (1 << (N3)));
				if(nextfree > 0)
					SQd2BlB(((ld|bit)<<2), ((rd|bit)>>2) | (1 << (N3)), col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd2BkBlB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQd2BlB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark2) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<2) | ((rd|bit)>>2) | (col|bit) | 1);
				if(nextfree > 0)
					SQd2B(((ld|bit)<<2) | 1, (rd|bit)>>2, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd2BlB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQd2BkB(int ld, int rd, int col, int idx, int free) {
		int bit;
		int nextfree;

		if(idx == mark2) {
			while(free > 0) {
				bit = free & (-free);
				free -= bit;
				nextfree = ~(((ld|bit)<<2) | ((rd|bit)>>2) | (col|bit) | L3);
				if(nextfree > 0)
					SQd2B(((ld|bit)<<2), ((rd|bit)>>2) | L3, col|bit, idx+1, nextfree);
			}
			return;
		}

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			nextfree = ~(((ld|bit)<<1) | ((rd|bit)>>1) | (col|bit));
			if(nextfree > 0)
				SQd2BkB((ld|bit)<<1, (rd|bit)>>1, col|bit, idx+1, nextfree);
		}
	}

	private void SQd2B(int ld, int rd, int col, int idx, int free) {
		if(idx == N5) {
			if((free & (~1)) > 0) 
				tempcounter++;
			return;
		}

		int bit;
		int nextfree;

		while(free > 0) {
			bit = free & (-free);
			free -= bit;
			
			int next_ld = ((ld|bit)<<1);
			int next_rd = ((rd|bit)>>1);
			int next_col = (col|bit);
			nextfree = ~(next_ld | next_rd | next_col);
			if(nextfree > 0)
				if(idx < N5-1) {
					if(~((next_ld<<1) | (next_rd>>1) | (next_col)) > 0)
						SQd2B(next_ld, next_rd, next_col, idx+1, nextfree);
				} else {
					SQd2B(next_ld, next_rd, next_col, idx+1, nextfree);
				}
		}
	}

	
	@Override
	public void run() {
		running = true;
		
		final int listsize = startConstellations.size();
		int i, j, k, l, ijkl, ld, rd, col;
		final int N = this.N, L = this.L;
		final int smallmask = (1 << (N-2)) - 1;
		
		for(int idx = 0; idx < listsize; idx++) {
			// apply jasmin and get i, j, k, l
			ijkl = startConstellations.getFirst();
			i = geti(ijkl); j = getj(ijkl); k = getk(ijkl); l = getl(ijkl);

			// big case distinction depending on distance d from queen j to right corner
			// d < 2
			if(N-1-j < 2) {
				// d = 1
				if(j != N-1) {
					// k < l
					if(k < l) {
						ld = (1 << (N-i-1)) | (L >> k);
						rd = (1 << (N-2)) | (L >> (i+2)) | (1 << (l-2));
						col = (1 << (N-2-i)) | 1;

						// 2 Blocks
						if(l == k+1) {
							mark2 = k - 2;
							SQd1BklB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
						}
						// 3 Blocks
						else {
							mark1 = k - 2;
							mark2 = l - 3;
							SQd1BkBlB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
						}
					}
					// l < k
					else {
						// l > 1
						if(l > 1) {
							ld = (1 << (N-i-1)) | (L >> k);
							rd = (1 << (N-2)) | (L >> (i+2)) | (1 << (l-2));
							col = (1 << (N-2-i)) | 1;

							// 1 Block, k and l are N-2 and N-3
							if(l == N-3) {
								SQd1B(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
							}
							// l < N-3
							else {
								// k not N-2
								if(k < N-2) {
									// 2 Blocks
									if(k == l+1) {
										mark2 = l - 2;
										SQd1BlkB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
									}
									// 3 Blocks
									else {
										mark1 = l - 2;
										mark2 = k - 3;
										SQd1BlBkB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
									}
								}
								// k = N-2, 2 Blocks
								else {
									mark2 = l - 2;
									SQd1BlB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
								}
							}

						}
						// l = 1
						else {
							ld = (1 << (N-i)) | (L >> (k-1)) | 1;
							rd = L3 | (L >> (i+3));
							col = (1 << (N-2-i)) | 1;

							// k = 2 , 1 Block
							if(k == 2) {
								ld <<= 1;
								rd >>= 1;
								rd |= L3;
								SQd1B(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
							}
							// k = N-2, 1 Block
							else if(k == N - 2) {
								SQd1B(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
							}
							// k in between, 2 Blocks
							else {
								mark2 = k - 3;
								SQd1BkB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
							}
						}
					}
				}
				// d = 0
				else {
					// 2 Blocks
					if(k != 1) {
						ld = (1 << (N-i-1)) | (L >> k);
						rd = (1 << (N-3)) | (L >> (i+2));
						col = (1 << (N-2-i));
						mark1 = k - 2;
						SQd0BkB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
					}
					// 1 Block
					else {
						ld = 1 << (N-i);
						rd = (1 << (N-4)) | (1 << (N-3)) | (L >> (i+3));
						col = (1 << (N-2-i));
						SQd0B(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
					}
				}
			}
			// N-1-j > 2
			else {
				// N-1-j = 2
				if(N-1-j == 2) {
					ld = (1 << (N-i-1)) | (L >> k);
					rd = (1 << (N-1-j+N-3)) | (L >> (i+2)) | (1 << (l-2));
					col = (1 << (N-2-i)) | (1 << (N-2-j));
					// k < l
					if(k < l) {
						mark1 = k - 2;
						mark2 = l - 3;
						// 2 Blocks
						if(l == k+1) 
							SQd2BklB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
						// 3 Blocks
						else 
							SQd2BkBlB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
					}
					// l < k
					else {
						mark1 = l - 2;
						mark2 = k - 3;
						// 2 Blocks
						if(k == l+1) 
							SQd2BlkB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
						// 3 Blocks
						else 
							SQd2BlBkB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
					}
				}
				// d > 2
				else {
					ld = (1 << (N-i-1)) | (L >> k);
					rd = (1 << (N-1-j+N-3)) | (L >> (i+2)) | (1 << (l-2));
					col = (1 << (N-2-i)) | (1 << (N-2-j));

					mark3 = j - 2;

					//k >= l
					if(k >= l) {
						mark1 = l - 2;
						mark2 = k - 3;
						// 2 Blocks
						if(k == l+1) 
							SQBlkBjrB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
						// 3 Blocks
						else 
							SQBlBkBjrB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
					}
					// l > k
					else {
						mark1 = k - 2;
						mark2 = l - 3;
						// 2 Blocks
						if(l == k+1) 
							SQBklBjrB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
						// 3 Blocks
						else 
							SQBkBlBjrB(ld, rd, col | (~smallmask), 0, (~(ld|rd|col)) & smallmask);
					}
				}
			}

			// sum up solutions
			solvecounter += tempcounter * symmetry(ijkl);

			// get occupancy of the board for each starting constellation and the hops and max from board Properties
			tempcounter = 0;								// set counter of solutions for this starting constellation to 0

			// for saving and loading progress remove the finished starting constellation
			startConstellations.removeFirst();
			
			// update the current startconstellation-index
			done++;
			
			// check for pausing
			if(pause == 1) {
				pause = 2;
				caller.onPauseStart();
				while(pause == 2) {
					if(cancel)
						break;
					try {
						Thread.sleep(100);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
			// check for cancelling
			if(cancel) {
				break;
			}
		}
		running = false;
	}

	// for user interaction
	void pauseThread() {
		pause = 1;
	}
	
	void cancelThread() {
		cancel = true;
	}
	
	void resumeThread() {
		pause = 0;
		cancel = false;
	}
	
	boolean isPaused() {
		return pause == 2;
	}
	
	boolean wasCanceled() {
		return !running && cancel;
	}
	
	// getters and setters
	int getDone() {
		return done;
	}
	
	long getSolutions() {
		return solvecounter;
	}
	
	ArrayDeque<Integer> getRemainingConstellations() {
		return startConstellations;
	}
	
	// helper functions for doing the math
	// for symmetry stuff and working with ijkl
	// true, if starting constellation is symmetric for rot90
	private boolean symmetry90(int ijkl) {
		if(((geti(ijkl) << 24) + (getj(ijkl) << 16) + (getk(ijkl) << 8) + getl(ijkl)) == (((N-1-getk(ijkl))<<24) + ((N-1-getl(ijkl))<<16) + (getj(ijkl)<<8) + geti(ijkl)))
			return true;
		return false;
	}
	// how often does a found solution count for this start constellation
	private int symmetry(int ijkl) {
		if(geti(ijkl) == N-1-getj(ijkl) && getk(ijkl) == N-1-getl(ijkl))		// starting constellation symmetric by rot180?
			if(symmetry90(ijkl))		// even by rot90?
				return 2;
			else
				return 4;
		else
			return 8;					// none of the above?
	}
	
	private int geti(int ijkl) {
		return ijkl >> 24;
	}
	private int getj(int ijkl) {
		return (ijkl >> 16) & 255;
	}
	private int getk(int ijkl) {
		return (ijkl >> 8) & 255;
	}
	private int getl(int ijkl) {
		return ijkl & 255;
	}
}

