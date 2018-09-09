#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"
#include <device_launch_parameters.h>

// 2.1 & 2.3 toggle between 1X & 2X grid size
#define GRID1X 1
#define GRID2X 0

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 1024

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

glm::vec3* dev_posBuffer;
glm::vec3* dev_vel1Buffer;
glm::vec3* dev_vel2Buffer;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
#if GRID2X
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
#endif // 2XGRID
#if GRID1X
  gridCellWidth = 1.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
#endif // 1XGRID

  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaDeviceSynchronize();
  // DO2.1
  // allocate memory for 
//  int *dev_particleArrayIndices; 
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArray failed!");
//  int *dev_particleGridIndices; 
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

//  int *dev_gridCellStartIndices; 
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");
//  int *dev_gridCellEndIndices;  
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  //DO2.3
  //allocate memory for shuffle buffers
  cudaMalloc((void**)&dev_posBuffer, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_posbuffer failed!");
  cudaMalloc((void**)&dev_vel1Buffer, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1buffer failed!");
  cudaMalloc((void**)&dev_vel2Buffer, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2buffer failed!");

}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO<<<fullBlocksPerGrid, blockSize >>>(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO<<<fullBlocksPerGrid, blockSize >>>(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids
  return glm::vec3(0.0f, 0.0f, 0.0f);
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	glm::vec3 vel_change(0.0f, 0.0f, 0.0f);
	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	if (index >= N) {
		return;
	}
	//the pos of current boid
	glm::vec3 thisPos = pos[index];
	//compute the perceived center of mass
	glm::vec3 perceived_CM(0.0f, 0.0f, 0.0f);
	float perceived_Num1 = 0.0f;
	// Rule 2: boids try to stay a distance d away from each other
	glm::vec3 c(0.0f, 0.0f, 0.0f);
	// Rule 3: boids try to match the speed of surrounding boids
	glm::vec3 perceived_Vel(0.0f, 0.0f, 0.0f);
	float perceived_Num3 = 0.0f;
	for (int index_i = 0; index_i < N; index_i++)
	{
		if (index_i != index)
		{
			glm::vec3 another_Pos = pos[index_i];
			float disFromThis = glm::distance(thisPos, another_Pos);
			if (disFromThis < rule1Distance)
			{
				perceived_CM += another_Pos;
				perceived_Num1 += 1.0f;
			}
			if (disFromThis < rule2Distance)
			{
				c -= (another_Pos - thisPos);
			}
			if (disFromThis < rule3Distance)
			{
				perceived_Vel += vel1[index_i];
				perceived_Num3 += 1.0f;
			}
		}
	}
	//rule1
	if (perceived_Num1 > 0)
	{
		perceived_CM = perceived_CM * (1.0f / perceived_Num1);
		vel_change += (perceived_CM - thisPos) * rule1Scale;
	}
	//rule2
	vel_change += c * rule2Scale;
	//rule3
	if (perceived_Num3 > 0)
	{
		perceived_Vel = perceived_Vel * (1.0f / perceived_Num3);
		vel_change += perceived_Vel * rule3Scale;
	}
  // Clamp the speed
	glm::vec3 newVel = vel1[index] + vel_change;
	newVel.x = newVel.x < -maxSpeed ? -maxSpeed : newVel.x;
	newVel.y = newVel.y < -maxSpeed ? -maxSpeed : newVel.y;
	newVel.z = newVel.z < -maxSpeed ? -maxSpeed : newVel.z;

	newVel.x = newVel.x > maxSpeed ? maxSpeed : newVel.x;
	newVel.y = newVel.y > maxSpeed ? maxSpeed : newVel.y;
	newVel.z = newVel.z > maxSpeed ? maxSpeed : newVel.z;
  // Record the new velocity into vel2. Question: why NOT vel1?
	vel2[index] = newVel; 
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}
	glm::vec3 currPos = pos[index];
	int x = (int)(currPos.x * inverseCellWidth) + gridResolution / 2;
	int y = (int)(currPos.y * inverseCellWidth) + gridResolution / 2;
	int z = (int)(currPos.z * inverseCellWidth) + gridResolution / 2;
	gridIndices[index] = gridIndex3Dto1D(x, y, z, gridResolution);
	indices[index] = index;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
	int indexThread = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (indexThread >= N)
	{
		return;
	}
	int gridIdx = particleGridIndices[indexThread];

	//the first 
	if (indexThread == 0)
	{
		gridCellStartIndices[gridIdx] = indexThread;
	}

	if (indexThread > 0)
	{
		int gridIdxPre = particleGridIndices[indexThread - 1];
		//this cell is dfferent from the previous cell, must be a new grid
		if (indexThread == N - 1)
		{
			gridCellEndIndices[gridIdx] = N - 1;
		}
		if (gridIdx != gridIdxPre )
		{
			gridCellStartIndices[gridIdx] = indexThread;
			gridCellEndIndices[gridIdxPre] = indexThread - 1;
		}
	}
	
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
	int gridstoSee[8];
	//neighboring cells
	int xflag, yflag, zflag;
	int boidPosVelInd = particleArrayIndices[index];
	glm::vec3 posCurr = pos[boidPosVelInd];
	int gridX = int(floor(posCurr.x * inverseCellWidth));
	int gridY = int(floor(posCurr.y * inverseCellWidth));
	int gridZ = int(floor(posCurr.z * inverseCellWidth));
	float xfloor = gridX * cellWidth;
	float yfloor = gridY * cellWidth;
	float zfloor = gridZ * cellWidth;

	float xDecimalRatio = (posCurr.x - xfloor) / cellWidth;
	float yDecimalRatio = (posCurr.y - yfloor) / cellWidth;
	float zDecimalRatio = (posCurr.z - zfloor) / cellWidth;

	int halfSideCount = gridResolution / 2;
	int gridSideCount = gridResolution;

	//the first grid is the current grid
	gridstoSee[0] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount, gridZ + halfSideCount, gridSideCount);
	//////////////x
	if (xDecimalRatio <= 0.5)
	{
		if (gridX >= -halfSideCount + 1)
		{
			gridstoSee[1] = gridIndex3Dto1D(gridX + halfSideCount - 1, gridY + halfSideCount, gridZ + halfSideCount, gridSideCount);
		}
		else
		{
			gridstoSee[1] = gridIndex3Dto1D(gridSideCount - 1, gridY + halfSideCount, gridZ + halfSideCount, gridSideCount);
		}
		xflag= 0;
	}
	else
	{
		if (gridX < halfSideCount-1)
		{
			gridstoSee[1] = gridIndex3Dto1D(gridX + halfSideCount + 1, gridY + halfSideCount, gridZ + halfSideCount, gridSideCount);
		}
		else
		{
			gridstoSee[1] = gridIndex3Dto1D(0, gridY + halfSideCount, gridZ + halfSideCount, gridSideCount);
		}
		xflag = 1;
	}
	////////////////y
	if (yDecimalRatio <= 0.5)
	{
		if (gridY >= -halfSideCount + 1)
		{
			gridstoSee[2] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount - 1, gridZ + halfSideCount, gridSideCount);
		}
		else
		{
			gridstoSee[2] = gridIndex3Dto1D(gridX + halfSideCount, gridSideCount - 1, gridZ + halfSideCount, gridSideCount);
		}
		yflag = 0;
	}
	else
	{
		if (gridY < halfSideCount - 1)
		{
			gridstoSee[2] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount + 1, gridZ + halfSideCount, gridSideCount);
		}
		else
		{
			gridstoSee[2] = gridIndex3Dto1D(gridX + halfSideCount, 0, gridZ + halfSideCount, gridSideCount);
		}
		yflag = 1;
	}
	/////////////////////z
	if (zDecimalRatio <= 0.5)
	{
		if (gridZ >= -halfSideCount + 1)
		{
			gridstoSee[3] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount, gridZ + halfSideCount - 1, gridSideCount);
		}
		else
		{
			gridstoSee[3] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount, gridSideCount - 1, gridSideCount);
		}
		zflag = 0;
	}
	else
	{
		if (gridZ < halfSideCount - 1)
		{
			gridstoSee[3] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount, gridZ + halfSideCount + 1, gridSideCount);
		}
		else
		{
			gridstoSee[3] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount, 0, gridSideCount);
		}
		zflag = 1;
	}
	// lets check for diagonal directions
	//xy

	float xyIndicator = (xDecimalRatio - xflag) * (xDecimalRatio - xflag) + (yDecimalRatio - yflag) * (yDecimalRatio - yflag);
	if (xyIndicator <= 0.25)
	{
		if (xflag == 0)
			xflag = -1;
		if (yflag == 0)
			yflag = -1;
		gridstoSee[4] = gridIndex3Dto1D((gridX + halfSideCount + xflag + gridSideCount) % gridSideCount, 
			(gridY + halfSideCount + yflag + gridSideCount) % gridSideCount, 
			gridZ + halfSideCount, gridSideCount);
	}
	else
		gridstoSee[4] = -1;
	//xz

	if (xflag < 0)
		xflag = 0;
	float xzIndicator = (xDecimalRatio - xflag) * (xDecimalRatio - xflag) + (zDecimalRatio - zflag) * (zDecimalRatio - zflag);
	if (xzIndicator <= 0.25)
	{
		if (!xflag)
			xflag = -1;
		if (!zflag)
			zflag = -1;
		gridstoSee[5] = gridIndex3Dto1D((gridX + halfSideCount + xflag + gridSideCount) % gridSideCount, gridY + halfSideCount, (gridZ + halfSideCount + zflag + gridSideCount) % gridSideCount, gridSideCount);
	}
	else
		gridstoSee[5] = -1;
	
	//yz

	if (yflag < 0)
		yflag = 0;
	if (zflag < 0)
		zflag = 0;
	float yzIndicator = (zDecimalRatio - zflag) * (zDecimalRatio - zflag) + (yDecimalRatio - yflag) * (yDecimalRatio - yflag);
	if (yzIndicator <= 0.25)
	{
		if (!zflag)
			zflag = -1;
		if (!yflag)
			yflag = -1;
		gridstoSee[6] = gridIndex3Dto1D(gridX + halfSideCount, (gridY + halfSideCount + yflag + gridSideCount) % gridSideCount, (gridZ + halfSideCount + zflag + gridSideCount) % gridSideCount, gridSideCount);
	}
	else
		gridstoSee[6] = -1;

	//xyz
	xflag = xflag < 0 ? 0 : xflag;
	yflag = yflag < 0 ? 0 : yflag;
	zflag = zflag < 0 ? 0 : zflag;
	float xyzIndicator = (xDecimalRatio - xflag) * (xDecimalRatio - xflag) 
		+ (yDecimalRatio - yflag) * (yDecimalRatio - yflag)
		+ (zDecimalRatio - zflag) * (zDecimalRatio - zflag);
	if (xyzIndicator <= 0.25)
	{
		if (!zflag)
			zflag = -1;
		if (!yflag)
			yflag = -1;
		if (!xflag)
			xflag = -1;
		gridstoSee[7] = gridIndex3Dto1D((gridX + halfSideCount + xflag + gridSideCount) % gridSideCount, (gridY + halfSideCount + yflag + gridSideCount) % gridSideCount, (gridZ + halfSideCount + zflag + gridSideCount) % gridSideCount, gridSideCount);

	}
	else
		gridstoSee[7] = -1;

  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
	glm::vec3 vel_change(0.0f, 0.0f, 0.0f);
	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	//compute the perceived center of mass
	glm::vec3 perceived_CM(0.0f, 0.0f, 0.0f);
	float perceived_Num1 = 0.0f;
	// Rule 2: boids try to stay a distance d away from each other
	glm::vec3 c(0.0f, 0.0f, 0.0f);
	// Rule 3: boids try to match the speed of surrounding boids
	glm::vec3 perceived_Vel(0.0f, 0.0f, 0.0f);
	float perceived_Num3 = 0.0f;

	for (int tocheckInd = 0; tocheckInd < 8; tocheckInd++)
	{
		int gridToCheck = gridstoSee[tocheckInd];
		if (gridToCheck >= 0)
		{
			//read the start/end indices 
			int startInd = gridCellStartIndices[gridToCheck];
			int endInd = gridCellEndIndices[gridToCheck];
			//access boids and compute velocity change
			//use boidPtrInd>=0 to surpass those cells with no boids inside

			for (int boidPtrInd = startInd; boidPtrInd <= endInd && boidPtrInd >= 0 ; boidPtrInd++)
			{
				int boidInd = particleArrayIndices[boidPtrInd];
				if (boidInd != boidPosVelInd)
				{
					glm::vec3 another_Pos = pos[boidInd];
					float disFromThis = glm::distance(posCurr, another_Pos);
					if (disFromThis < rule1Distance)
					{
						perceived_CM += another_Pos;
						perceived_Num1 += 1.0f;
					}
					if (disFromThis < rule2Distance)
					{
						c -= (another_Pos - posCurr);
					}
					if (disFromThis < rule3Distance)
					{
						perceived_Vel += vel1[boidInd];
						perceived_Num3 += 1.0f;
					}
				}	
			}
		}
	}
	//rule1
	if (perceived_Num1 > 0)
	{
		perceived_CM = perceived_CM * (1.0f / perceived_Num1);
		vel_change += (perceived_CM - posCurr) * rule1Scale;
	}
	//rule2
	vel_change += c * rule2Scale;
	//rule3
	if (perceived_Num3 > 0)
	{
		perceived_Vel = perceived_Vel * (1.0f / perceived_Num3);
		vel_change += perceived_Vel * rule3Scale;
	}
	glm::vec3 newVel = vel1[boidPosVelInd] + vel_change;
  // - Clamp the speed change before putting the new speed in vel2
	newVel.x = newVel.x < -maxSpeed ? -maxSpeed : newVel.x;
	newVel.y = newVel.y < -maxSpeed ? -maxSpeed : newVel.y;
	newVel.z = newVel.z < -maxSpeed ? -maxSpeed : newVel.z;

	newVel.x = newVel.x > maxSpeed ? maxSpeed : newVel.x;
	newVel.y = newVel.y > maxSpeed ? maxSpeed : newVel.y;
	newVel.z = newVel.z > maxSpeed ? maxSpeed : newVel.z;
	
	vel2[boidPosVelInd] = newVel;
}

__global__ void kernUpdateVelNeighborSearchScattered1XGRID(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int *gridCellStartIndices, int *gridCellEndIndices,
	int *particleArrayIndices,
	glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	// This is for 1xGRID case & scattered position
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}
	// - Identify the grid cell that this particle is in
	// - Identify which cells may contain neighbors. In this case there are 27 cells
	int halfSideCount = gridResolution / 2;
	int gridSideCount = gridResolution;
	int boidPosVelInd = particleArrayIndices[index];
	glm::vec3 posCurr = pos[boidPosVelInd];
	int gridX = int(floor(posCurr.x * inverseCellWidth)) + halfSideCount;
	int gridY = int(floor(posCurr.y * inverseCellWidth)) + halfSideCount;
	int gridZ = int(floor(posCurr.z * inverseCellWidth)) + halfSideCount;
	//  Then the current boid is located inside the grid of (gridX, gridY, gridZ)
	//  Just need to check the surrounding 3x3x3 cells
	// - Access each boid in the cell and compute velocity change from
	//   the boids rules, if this boid is within the neighborhood distance.
	glm::vec3 vel_change(0.0f, 0.0f, 0.0f);
	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	//compute the perceived center of mass
	glm::vec3 perceived_CM(0.0f, 0.0f, 0.0f);
	float perceived_Num1 = 0.0f;
	// Rule 2: boids try to stay a distance d away from each other
	glm::vec3 c(0.0f, 0.0f, 0.0f);
	// Rule 3: boids try to match the speed of surrounding boids
	glm::vec3 perceived_Vel(0.0f, 0.0f, 0.0f);
	float perceived_Num3 = 0.0f;
	//debugging

	for (int zDir = -1; zDir <= 1; zDir++)
	{
		for (int yDir = -1; yDir <= 1; yDir++)
		{
			for (int xDir = -1; xDir <= 1; xDir++)
			{
				//  use module to get desired position
				//  compute the grid index
				int gridInd = gridIndex3Dto1D((gridX + xDir + gridSideCount) % gridSideCount,
					(gridY + yDir + gridSideCount) % gridSideCount,
					(gridZ + zDir + gridSideCount) % gridSideCount,
					gridSideCount);
				//read the start/end indices 
				int startInd = gridCellStartIndices[gridInd];
				int endInd = gridCellEndIndices[gridInd];

				//access boids and compute velocity change
				//use boidPtrInd>=0 to surpass those cells with no boids inside

				for (int boidPtrInd = startInd; boidPtrInd <= endInd && boidPtrInd >= 0; boidPtrInd++)
				{
					int boidInd = particleArrayIndices[boidPtrInd];
					/////following is basically copy from 1.2
					if (boidInd != boidPosVelInd)
					{
						glm::vec3 another_Pos = pos[boidInd];
						float disFromThis = glm::distance(posCurr, another_Pos);
						if (disFromThis < rule1Distance)
						{
							perceived_CM += another_Pos;
							perceived_Num1 += 1.0f;
						}
						if (disFromThis < rule2Distance)
						{
							c -= (another_Pos - posCurr);
						}
						if (disFromThis < rule3Distance)
						{
							perceived_Vel += vel1[boidInd];
							perceived_Num3 += 1.0f;
						}
					}
				}
			}
		}
	}
	//rule1
	if (perceived_Num1 > 0)
	{
		perceived_CM = perceived_CM * (1.0f / perceived_Num1);
		vel_change += (perceived_CM - posCurr) * rule1Scale;
	}
	//rule2
	vel_change += c * rule2Scale;
	//rule3
	if (perceived_Num3 > 0)
	{
		perceived_Vel = perceived_Vel * (1.0f / perceived_Num3);
		vel_change += perceived_Vel * rule3Scale;
	}

	//debugging
	glm::vec3 newVel = vel1[boidPosVelInd] + vel_change;
	//glm::vec3 newVel = vel1[boidPosVelInd];
	// - Clamp the speed change before putting the new speed in vel2
	newVel.x = newVel.x < -maxSpeed ? -maxSpeed : newVel.x;
	newVel.y = newVel.y < -maxSpeed ? -maxSpeed : newVel.y;
	newVel.z = newVel.z < -maxSpeed ? -maxSpeed : newVel.z;

	newVel.x = newVel.x > maxSpeed ? maxSpeed : newVel.x;
	newVel.y = newVel.y > maxSpeed ? maxSpeed : newVel.y;
	newVel.z = newVel.z > maxSpeed ? maxSpeed : newVel.z;

	vel2[boidPosVelInd] = newVel;
	
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}
	// - Identify the grid cell that this particle is in
	// - Identify which cells may contain neighbors. This isn't always 8.
	int gridstoSee[8];
	//neighboring cells
	int xflag, yflag, zflag;
	int boidPosVelInd = index;
	glm::vec3 posCurr = pos[boidPosVelInd];
	int gridX = int(floor(posCurr.x * inverseCellWidth));
	int gridY = int(floor(posCurr.y * inverseCellWidth));
	int gridZ = int(floor(posCurr.z * inverseCellWidth));
	float xfloor = gridX * cellWidth;
	float yfloor = gridY * cellWidth;
	float zfloor = gridZ * cellWidth;

	float xDecimalRatio = (posCurr.x - xfloor) / cellWidth;
	float yDecimalRatio = (posCurr.y - yfloor) / cellWidth;
	float zDecimalRatio = (posCurr.z - zfloor) / cellWidth;

	int halfSideCount = gridResolution / 2;
	int gridSideCount = gridResolution;

	//the first grid is the current grid
	gridstoSee[0] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount, gridZ + halfSideCount, gridSideCount);


	//////////////x
	if (xDecimalRatio <= 0.5)
	{
		if (gridX >= -halfSideCount + 1)
		{
			gridstoSee[1] = gridIndex3Dto1D(gridX + halfSideCount - 1, gridY + halfSideCount, gridZ + halfSideCount, gridSideCount);
		}
		else
		{
			gridstoSee[1] = gridIndex3Dto1D(gridSideCount - 1, gridY + halfSideCount, gridZ + halfSideCount, gridSideCount);
		}
		xflag = 0;
	}
	else
	{
		if (gridX < halfSideCount - 1)
		{
			gridstoSee[1] = gridIndex3Dto1D(gridX + halfSideCount + 1, gridY + halfSideCount, gridZ + halfSideCount, gridSideCount);
		}
		else
		{
			gridstoSee[1] = gridIndex3Dto1D(0, gridY + halfSideCount, gridZ + halfSideCount, gridSideCount);
		}
		xflag = 1;
	}
	//y
	if (yDecimalRatio <= 0.5)
	{
		if (gridY >= -halfSideCount + 1)
		{
			gridstoSee[2] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount - 1, gridZ + halfSideCount, gridSideCount);
		}
		else
		{
			gridstoSee[2] = gridIndex3Dto1D(gridX + halfSideCount, gridSideCount - 1, gridZ + halfSideCount, gridSideCount);
		}
		yflag = 0;
	}
	else
	{
		if (gridY < halfSideCount - 1)
		{
			gridstoSee[2] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount + 1, gridZ + halfSideCount, gridSideCount);
		}
		else
		{
			gridstoSee[2] = gridIndex3Dto1D(gridX + halfSideCount, 0, gridZ + halfSideCount, gridSideCount);
		}
		yflag = 1;
	}
	//z
	if (zDecimalRatio <= 0.5)
	{
		if (gridZ >= -halfSideCount + 1)
		{
			gridstoSee[3] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount, gridZ + halfSideCount - 1, gridSideCount);
		}
		else
		{
			gridstoSee[3] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount, gridSideCount - 1, gridSideCount);
		}
		zflag = 0;
	}
	else
	{
		if (gridZ < halfSideCount - 1)
		{
			gridstoSee[3] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount, gridZ + halfSideCount + 1, gridSideCount);
		}
		else
		{
			gridstoSee[3] = gridIndex3Dto1D(gridX + halfSideCount, gridY + halfSideCount, 0, gridSideCount);
		}
		zflag = 1;
	}
	// lets check for diagonal directions
	//xy
	float xyIndicator = (xDecimalRatio - xflag) * (xDecimalRatio - xflag) + (yDecimalRatio - yflag) * (yDecimalRatio - yflag);
	if (xyIndicator <= 0.25)
	{
		if (xflag == 0)
			xflag = -1;
		if (yflag == 0)
			yflag = -1;
		gridstoSee[4] = gridIndex3Dto1D((gridX + halfSideCount + xflag + gridSideCount) % gridSideCount,
			(gridY + halfSideCount + yflag + gridSideCount) % gridSideCount,
			gridZ + halfSideCount, gridSideCount);
	}
	else
		gridstoSee[4] = -1;
	//xz
	if (xflag < 0)
		xflag = 0;
	float xzIndicator = (xDecimalRatio - xflag) * (xDecimalRatio - xflag) + (zDecimalRatio - zflag) * (zDecimalRatio - zflag);
	if (xzIndicator <= 0.25)
	{
		if (!xflag)
			xflag = -1;
		if (!zflag)
			zflag = -1;
		gridstoSee[5] = gridIndex3Dto1D((gridX + halfSideCount + xflag + gridSideCount) % gridSideCount, gridY + halfSideCount, (gridZ + halfSideCount + zflag + gridSideCount) % gridSideCount, gridSideCount);
	}
	else
		gridstoSee[5] = -1;

	//yz
	if (yflag < 0)
		yflag = 0;
	if (zflag < 0)
		zflag = 0;
	float yzIndicator = (zDecimalRatio - zflag) * (zDecimalRatio - zflag) + (yDecimalRatio - yflag) * (yDecimalRatio - yflag);
	if (yzIndicator <= 0.25)
	{
		if (!zflag)
			zflag = -1;
		if (!yflag)
			yflag = -1;
		gridstoSee[6] = gridIndex3Dto1D(gridX + halfSideCount, (gridY + halfSideCount + yflag + gridSideCount) % gridSideCount, (gridZ + halfSideCount + zflag + gridSideCount) % gridSideCount, gridSideCount);
	}
	else
		gridstoSee[6] = -1;

	//xyz
	xflag = xflag < 0 ? 0 : xflag;
	yflag = yflag < 0 ? 0 : yflag;
	zflag = zflag < 0 ? 0 : zflag;
	float xyzIndicator = (xDecimalRatio - xflag) * (xDecimalRatio - xflag)
		+ (yDecimalRatio - yflag) * (yDecimalRatio - yflag)
		+ (zDecimalRatio - zflag) * (zDecimalRatio - zflag);
	if (xyzIndicator <= 0.25)
	{
		if (!zflag)
			zflag = -1;
		if (!yflag)
			yflag = -1;
		if (!xflag)
			xflag = -1;
		gridstoSee[7] = gridIndex3Dto1D((gridX + halfSideCount + xflag + gridSideCount) % gridSideCount, (gridY + halfSideCount + yflag + gridSideCount) % gridSideCount, (gridZ + halfSideCount + zflag + gridSideCount) % gridSideCount, gridSideCount);

	}
	else
		gridstoSee[7] = -1;
	// - For each cell, read the start/end indices in the boid pointer array.
	//   DIFFERENCE: For best results, consider what order the cells should be
	//   checked in to maximize the memory benefits of reordering the boids data.
    //     for(z)
	//		 for(y)
	//			 for(x)
	//   would be fastest according to the way we compute 1D indices from 3D indices
	//   here to maximize performance we need to rearrange the gridstoSee array
	//   this would be clearer in 1x cell size case
	int temp = 0;
	temp = gridstoSee[3];
	gridstoSee[3] = gridstoSee[4];
	gridstoSee[4] = temp;
	//   after rearrangement, it would be : self,x,y,xy,z,zx,yz,xyz which is the unrolled sequence of zyx-loop
	// - Access each boid in the cell and compute velocity change from
	//   the boids rules, if this boid is within the neighborhood distance.
	glm::vec3 vel_change(0.0f, 0.0f, 0.0f);
	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	//compute the perceived center of mass
	glm::vec3 perceived_CM(0.0f, 0.0f, 0.0f);
	float perceived_Num1 = 0.0f;
	// Rule 2: boids try to stay a distance d away from each other
	glm::vec3 c(0.0f, 0.0f, 0.0f);
	// Rule 3: boids try to match the speed of surrounding boids
	glm::vec3 perceived_Vel(0.0f, 0.0f, 0.0f);
	float perceived_Num3 = 0.0f;

	for (int tocheckInd = 0; tocheckInd < 8; tocheckInd++)
	{
		int gridToCheck = gridstoSee[tocheckInd];
		if (gridToCheck >= 0)
		{
			//read the start/end indices 
			int startInd = gridCellStartIndices[gridToCheck];
			int endInd = gridCellEndIndices[gridToCheck];
			//access boids and compute velocity change
			//use boidPtrInd>=0 to surpass those cells with no boids inside

			for (int boidPtrInd = startInd; boidPtrInd <= endInd && boidPtrInd >= 0; boidPtrInd++)
			{
				int boidInd = boidPtrInd;
				/////following is basically copy from 1.2
				if (boidInd != boidPosVelInd)
				{
					glm::vec3 another_Pos = pos[boidInd];
					float disFromThis = glm::distance(posCurr, another_Pos);
					if (disFromThis < rule1Distance)
					{
						perceived_CM += another_Pos;
						perceived_Num1 += 1.0f;
					}
					if (disFromThis < rule2Distance)
					{
						c -= (another_Pos - posCurr);
					}
					if (disFromThis < rule3Distance)
					{
						perceived_Vel += vel1[boidInd];
						perceived_Num3 += 1.0f;
					}
				}
			}
		}
	}
	//rule1
	if (perceived_Num1 > 0)
	{
		perceived_CM = perceived_CM * (1.0f / perceived_Num1);
		vel_change += (perceived_CM - posCurr) * rule1Scale;
	}
	//rule2
	vel_change += c * rule2Scale;
	//rule3
	if (perceived_Num3 > 0)
	{
		perceived_Vel = perceived_Vel * (1.0f / perceived_Num3);
		vel_change += perceived_Vel * rule3Scale;
	}

	glm::vec3 newVel = vel1[boidPosVelInd] + vel_change;
	// - Clamp the speed change before putting the new speed in vel2
	newVel.x = newVel.x < -maxSpeed ? -maxSpeed : newVel.x;
	newVel.y = newVel.y < -maxSpeed ? -maxSpeed : newVel.y;
	newVel.z = newVel.z < -maxSpeed ? -maxSpeed : newVel.z;

	newVel.x = newVel.x > maxSpeed ? maxSpeed : newVel.x;
	newVel.y = newVel.y > maxSpeed ? maxSpeed : newVel.y;
	newVel.z = newVel.z > maxSpeed ? maxSpeed : newVel.z;

	vel2[boidPosVelInd] = newVel;
}


//  1X Grid Size Case, use this function
__global__ void kernUpdateVelNeighborSearchCoherent1XGRID(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int *gridCellStartIndices, int *gridCellEndIndices,
	glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	// If the gridcellwidth is 1x search distance, we need to check 27 cells
	// the computation process is actually easier than 2x grid case
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}
	// - Identify the grid cell that this particle is in
	// - Identify which cells may contain neighbors. Here there are 27 cells
	int halfSideCount = gridResolution / 2;
	int gridSideCount = gridResolution;
	glm::vec3 posCurr = pos[index];
	int gridX = int(floor(posCurr.x * inverseCellWidth)) + halfSideCount;
	int gridY = int(floor(posCurr.y * inverseCellWidth)) + halfSideCount;
	int gridZ = int(floor(posCurr.z * inverseCellWidth)) + halfSideCount;
	//  Then the current boid is located inside the grid of (gridX, gridY, gridZ)
	//  Just need to check the surrounding 3x3x3 cells
	// - Access each boid in the cell and compute velocity change from
	//   the boids rules, if this boid is within the neighborhood distance.
	glm::vec3 vel_change(0.0f, 0.0f, 0.0f);
	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	//compute the perceived center of mass
	glm::vec3 perceived_CM(0.0f, 0.0f, 0.0f);
	float perceived_Num1 = 0.0f;
	// Rule 2: boids try to stay a distance d away from each other
	glm::vec3 c(0.0f, 0.0f, 0.0f);
	// Rule 3: boids try to match the speed of surrounding boids
	glm::vec3 perceived_Vel(0.0f, 0.0f, 0.0f);
	float perceived_Num3 = 0.0f;

	for (int zDir = -1; zDir <= 1; zDir++)
	{
		for (int yDir = -1; yDir <= 1; yDir++)
		{
			for (int xDir = -1; xDir <= 1; xDir++)
			{
				//  use module to get desired position
				//  compute the grid index
				int gridInd = gridIndex3Dto1D((gridX + xDir + gridSideCount) % gridSideCount,
					(gridY + yDir + gridSideCount) % gridSideCount,
					(gridZ + zDir + gridSideCount) % gridSideCount,
					gridSideCount);
				//read the start/end indices 
				int startInd = gridCellStartIndices[gridInd];
				int endInd = gridCellEndIndices[gridInd];

				//access boids and compute velocity change
				//use boidPtrInd>=0 to surpass those cells with no boids inside

				for (int boidPtrInd = startInd; boidPtrInd <= endInd && boidPtrInd >= 0; boidPtrInd++)
				{
					int boidInd = boidPtrInd;
					/////following is basically copy from 1.2
					if (boidInd != index)
					{
						glm::vec3 another_Pos = pos[boidInd];
						float disFromThis = glm::distance(posCurr, another_Pos);
						if (disFromThis < rule1Distance)
						{
							perceived_CM += another_Pos;
							perceived_Num1 += 1.0f;
						}
						if (disFromThis < rule2Distance)
						{
							c -= (another_Pos - posCurr);
						}
						if (disFromThis < rule3Distance)
						{
							perceived_Vel += vel1[boidInd];
							perceived_Num3 += 1.0f;
						}
					}
				}
			}
		}
	}
	//rule1
	if (perceived_Num1 > 0)
	{
		perceived_CM = perceived_CM * (1.0f / perceived_Num1);
		vel_change += (perceived_CM - posCurr) * rule1Scale;
	}
	//rule2
	vel_change += c * rule2Scale;
	//rule3
	if (perceived_Num3 > 0)
	{
		perceived_Vel = perceived_Vel * (1.0f / perceived_Num3);
		vel_change += perceived_Vel * rule3Scale;
	}

	glm::vec3 newVel = vel1[index] + vel_change;
	// - Clamp the speed change before putting the new speed in vel2
	newVel.x = newVel.x < -maxSpeed ? -maxSpeed : newVel.x;
	newVel.y = newVel.y < -maxSpeed ? -maxSpeed : newVel.y;
	newVel.z = newVel.z < -maxSpeed ? -maxSpeed : newVel.z;

	newVel.x = newVel.x > maxSpeed ? maxSpeed : newVel.x;
	newVel.y = newVel.y > maxSpeed ? maxSpeed : newVel.y;
	newVel.z = newVel.z > maxSpeed ? maxSpeed : newVel.z;

	vel2[index] = newVel;
}


/**
* ping-pong speed buffer
*/
__global__ void SwapSpeed(int N, glm::vec3 *vel1, glm::vec3 *vel2)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	vel1[index].x = vel2[index].x;
	vel1[index].y = vel2[index].y;
	vel1[index].z = vel2[index].z;
	return;
}
/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
 // // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	kernUpdateVelocityBruteForce <<<fullBlocksPerGrid, blockSize >>> (numObjects, dev_pos, dev_vel1, dev_vel2);
	kernUpdatePos <<<fullBlocksPerGrid, blockSize >>> (numObjects, dt, dev_pos, dev_vel2);
 // // TODO-1.2 ping-pong the velocity buffers
	SwapSpeed<<<fullBlocksPerGrid, blockSize >>>(numObjects, dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
	kernComputeIndices<<<fullBlocksPerGrid, blockSize >>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
	dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
	dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices

   //before we do this, need to set default values for start and end index:
	dim3 fullBlocksPerGrid2((gridCellCount + blockSize - 1) / blockSize);
	kernResetIntBuffer<<<fullBlocksPerGrid2,blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <fullBlocksPerGrid2, blockSize >> >(gridCellCount, dev_gridCellEndIndices, -1);
    //then compute the start and end indices
	kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize >>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  // - Perform velocity updates using neighbor search
#if GRID1X
	kernUpdateVelNeighborSearchScattered1XGRID << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
#endif

#if GRID2X
	kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
#endif  
	// - Update positions
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	// - Ping-pong buffers as needed
	SwapSpeed << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, dev_vel2);
}

// funtion to rearrange values
__global__ void shuffleBufferWithIndices(int N, bool back, glm::vec3* originalData1, glm::vec3* shuffledData1, glm::vec3* originalData2, glm::vec3* shuffledData2, int* particleArrayIndices)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}
	int realPos = particleArrayIndices[index];
	if (!back)
	{
		shuffledData1[index] = originalData1[realPos];
		shuffledData2[index] = originalData2[realPos];
	}
	else
	{
		originalData1[realPos] = shuffledData1[index];
		originalData2[realPos] = shuffledData2[index];
	}
}


void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
	kernComputeIndices<<<fullBlocksPerGrid,blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
	dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
	dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // Note - before we do this, need to set default values for start and end index:
	dim3 fullBlocksPerGrid2((gridCellCount + blockSize - 1) / blockSize);
	kernResetIntBuffer << <fullBlocksPerGrid2, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <fullBlocksPerGrid2, blockSize >> >(gridCellCount, dev_gridCellEndIndices, -1);
  // then compute the start and end indices
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
	shuffleBufferWithIndices<<<fullBlocksPerGrid,blockSize>>>(numObjects, false, dev_pos, dev_posBuffer, dev_vel1, dev_vel1Buffer, dev_particleArrayIndices);
	//shuffleBufferWithIndices(numObjects, dev_vel2, dev_vel2Buffer, dev_particleArrayIndices);
  // - Perform velocity updates using neighbor search
#if GRID1X
	kernUpdateVelNeighborSearchCoherent1XGRID << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices, dev_posBuffer, dev_vel1Buffer, dev_vel2Buffer);
#endif

#if GRID2X
	kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices, dev_posBuffer, dev_vel1Buffer, dev_vel2Buffer);
#endif
  // - Update positions
	kernUpdatePos<<<fullBlocksPerGrid,blockSize>>>(numObjects, dt, dev_posBuffer, dev_vel2Buffer);
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
	SwapSpeed<<<fullBlocksPerGrid,blockSize>>>(numObjects, dev_vel1Buffer, dev_vel2Buffer);
	// need to put pos and vel1 back to original position
	shuffleBufferWithIndices << <fullBlocksPerGrid, blockSize >> >(numObjects, true, dev_pos, dev_posBuffer, dev_vel1, dev_vel1Buffer, dev_particleArrayIndices);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_gridCellStartIndices);

  cudaFree(dev_posBuffer);
  cudaFree(dev_vel1Buffer);
  cudaFree(dev_vel2Buffer);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
