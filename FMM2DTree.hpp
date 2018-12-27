#ifndef _FMM2DTree_HPP__
#define _FMM2DTree_HPP__

#include <vector>
#include <Eigen/Dense>
#include <fstream>

#define EIGEN_DONT_PARALLELIZE
#include <iostream>
#include <cmath>

using namespace std;

const double PI	=	3.1415926535897932384;

struct pts2D {
	double x,y;
};

struct charge {
	double q,x,y;
};

struct level_box {
	int level,box;
};

class FMM2DBox {
public:
	bool exists;
	int boxNumber;
	int parentNumber;
	int childrenNumbers[4];
	int neighborNumbers[8];
	int innerNumbers[16];
	int outerNumbers[24];

	std::vector<level_box> list1;		//	Neighbors or boxes that share a boundary
	//std::vector<level_box> list2;		//	Interactions at same level
	std::vector<level_box> list3;		//	Descendants of neighbors at the same level, which are not neighbors
	std::vector<level_box> list4;		//	Opposite of List 3

	FMM2DBox () {
		boxNumber		=	-1;
		parentNumber	=	-1;
		for (int l=0; l<4; ++l) {
			childrenNumbers[l]	=	-1;
		}
		for (int l=0; l<8; ++l) {
			neighborNumbers[l]	=	-1;
		}
		for (int l=0; l<16; ++l) {
			innerNumbers[l]		=	-1;
		}
		for (int l=0; l<24; ++l) {
			outerNumbers[l]		=	-1;
		}		
	}
	//	charge database of charges present in the box
	std::vector<int> charge_indices;
  	
	Eigen::VectorXd multipoles;
	Eigen::VectorXd locals;

	pts2D center;

	//	The following will be stored only at the leaf nodes
	std::vector<pts2D> chebNodes;
};

class kernel {
public:
	bool isTrans;		//	Checks if the kernel is translation invariant, i.e., the kernel is K(r).
	bool isHomog;		//	Checks if the kernel is homogeneous, i.e., K(r) = r^{alpha}.
	bool isLogHomog;	//	Checks if the kernel is log-homogeneous, i.e., K(r) = log(r^{alpha}).
	double alpha;		//	Degree of homogeneity of the kernel.
	kernel() {};
	~kernel() {};
	virtual double getInteraction(const pts2D r1, const pts2D r2, double a){
		return 0.0;
	};	//	Kernel entry generator
};

template <typename kerneltype>
class FMM2DTree {
public:
	kerneltype* K;
	int nLevels;			//	Number of levels in the tree.
	int nChebNodes;			//	Number of Chebyshev nodes along one direction.
	int rank;				//	Rank of interaction, i.e., rank = nChebNodes*nChebNodes.
	int N;					//	Number of particles.
	double L;				//	Semi-length of the simulation box.
	double smallestBoxSize;	//	This is L/2.0^(nLevels).
	double a;				//	Cut-off for self-interaction. This is less than the length of the smallest box size.

	int MIN_CHARGES_PER_BOX;//	Minimum number of charges in one box

	std::vector<int> nBoxesPerLevel;			//	Number of boxes at each level in the tree.
	std::vector<double> boxRadius;				//	Box radius at each level in the tree assuming the box at the root is [-1,1]^2
	std::vector<double> boxHomogRadius;			//	Stores the value of boxRadius^{alpha}
	std::vector<double> boxLogHomogRadius;		//	Stores the value of alpha*log(boxRadius)
	std::vector<std::vector<FMM2DBox> > tree;	//	The tree storing all the information.
	
	//	childless_boxes
	std::vector<level_box> childless_boxes;

	//	charge in coloumbs and its location
	std::vector<charge> charge_database;
  	
	//	Chebyshev nodes
	std::vector<double> standardChebNodes1D;
	std::vector<pts2D> standardChebNodes;
	std::vector<pts2D> standardChebNodesChild;
	std::vector<pts2D> leafChebNodes;

	//	Different Operators
	Eigen::MatrixXd selfInteraction;		//	Needed only at the leaf level.
	Eigen::MatrixXd neighborInteraction[8];	//	Neighbor interaction only needed at the leaf level.
	Eigen::MatrixXd M2M[4];					//	Transfer from multipoles of 4 children to multipoles of parent.
	Eigen::MatrixXd L2L[4];					//	Transfer from locals of parent to locals of 4 children.
	Eigen::MatrixXd M2LInner[16];			//	M2L of inner interactions. This is done on the box [-L,L]^2.
	Eigen::MatrixXd M2LOuter[24];			//	M2L of outer interactions. This is done on the box [-L,L]^2.

	//	the potential evaluated by interpolating the potential at the locals and adding the near field
	Eigen::VectorXd potential;

// public:
	FMM2DTree(kerneltype* K, /*int nLevels,*/ int nChebNodes, double L) {
		this->K						=	K;
		//this->nLevels			=	nLevels;
		this->nChebNodes			=	nChebNodes;
		this->rank					=	nChebNodes*nChebNodes;
		this->L						=	L;
		this->MIN_CHARGES_PER_BOX	=	4*rank;
	}

	std::vector<pts2D> shift_Cheb_Nodes(double xShift, double yShift) {
		std::vector<pts2D> shiftedChebNodes;
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	standardChebNodes[k].x+2*xShift;
			temp.y	=	standardChebNodes[k].y+2*yShift;
			shiftedChebNodes.push_back(temp);
		}
		return shiftedChebNodes;
	}
	
	std::vector<pts2D> shift_Leaf_Cheb_Nodes(double xShift, double yShift) {
		std::vector<pts2D> shiftedChebNodes;
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	leafChebNodes[k].x+xShift;
			temp.y	=	leafChebNodes[k].y+yShift;
			shiftedChebNodes.push_back(temp);
		}
		return shiftedChebNodes;
	}


	//	shifted_scaled_cheb_nodes	//	used in evaluating multipoles
	std::vector<pts2D> shift_scale_Cheb_Nodes(double xShift, double yShift, double radius) {
		std::vector<pts2D> shifted_scaled_ChebNodes;
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	radius*standardChebNodes[k].x+xShift;
			temp.y	=	radius*standardChebNodes[k].y+yShift;
			shifted_scaled_ChebNodes.push_back(temp);
		}
		return shifted_scaled_ChebNodes;
	}


	//	get_ChebPoly
	double get_ChebPoly(double x, int n) {
		return cos(n*acos(x));
	}

	//	get_S
	double get_S(double x, double y, int n) {
		double S	=	0.5;
		for (int k=1; k<n; ++k) {
			S+=get_ChebPoly(x,k)*get_ChebPoly(y,k);
		}
		return 2.0/n*S;
	}
	//	set_Standard_Cheb_Nodes
	void set_Standard_Cheb_Nodes() {
		for (int k=0; k<nChebNodes; ++k) {
			standardChebNodes1D.push_back(-cos((k+0.5)/nChebNodes*PI));
		}
		pts2D temp1;
		for (int j=0; j<nChebNodes; ++j) {
			for (int k=0; k<nChebNodes; ++k) {
				temp1.x	=	standardChebNodes1D[k];
				temp1.y	=	standardChebNodes1D[j];
				standardChebNodes.push_back(temp1);
			}
		}
		//	Left Bottom child, i.e., Child 0
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x-0.5;
				temp1.y	=	0.5*temp1.y-0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Right Bottom child, i.e., Child 1
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x+0.5;
				temp1.y	=	0.5*temp1.y-0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Right Top child, i.e., Child 2
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x+0.5;
				temp1.y	=	0.5*temp1.y+0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Left Top child, i.e., Child 3
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x-0.5;
				temp1.y	=	0.5*temp1.y+0.5;
				standardChebNodesChild.push_back(temp1);
		}
	}

	void get_Transfer_Matrix() {
		for (int l=0; l<4; ++l) {
			L2L[l]	=	Eigen::MatrixXd(rank,rank);
			for (int j=0; j<rank; ++j) {
				for (int k=0; k<rank; ++k) {
					L2L[l](j,k)	=	get_S(standardChebNodes[k].x, standardChebNodesChild[j+l*rank].x, nChebNodes)*get_S(standardChebNodes[k].y, standardChebNodesChild[j+l*rank].y, nChebNodes);
				}
			}
		}
		for (int l=0; l<4; ++l) {
			M2M[l]	=	L2L[l].transpose();
		}
	}


	

	void read_inputs(string fileName) {
	//reading charge data from a file
		ifstream myfile;
		myfile.open (fileName.c_str());
		charge a;
  		double b;
  		int c =0;
  		while (myfile >> b){
			if(c==0){
				a.q	=	b;
				c = 1;
			}
			else if(c==1){
				a.x	=	b;
				c = 2;
			}
			else{
				a.y	=	b;
				c=0;
				charge_database.push_back(a);
			}
	        }
  		myfile.close();
		
	}

	void createTree() {
		//	First create root and add to tree
		FMM2DBox root;
		root.exists	=	true;
		root.boxNumber		=	0;
		root.parentNumber	=	-1;
		//not sure if it has children
		/*
		#pragma omp parallel for
		for (int l=0; l<4; ++l) {
			root.childrenNumbers[l]	=	l;
		}
		*/
		#pragma omp parallel for
		for (int l=0; l<8; ++l) {
			root.neighborNumbers[l]	=	-1;
		}
		#pragma omp parallel for
		for (int l=0; l<16; ++l) {
			root.innerNumbers[l]	=	-1;
		}
		#pragma omp parallel for
		for (int l=0; l<24; ++l) {
			root.outerNumbers[l]	=	-1;
		}
		for (int j=0; j<charge_database.size(); j++) {
			root.charge_indices.push_back(j);
		}
		root.center.x = 0.0;
		root.center.y = 0.0;
		
		std::vector<FMM2DBox> rootLevel;
		rootLevel.push_back(root);
		tree.push_back(rootLevel);
		
		nBoxesPerLevel.push_back(1);
		boxRadius.push_back(L);
		boxHomogRadius.push_back(pow(L,K->alpha));
		boxLogHomogRadius.push_back(K->alpha*log(L));

		int j=1;
		while (1) {
			nBoxesPerLevel.push_back(4*nBoxesPerLevel[j-1]);
			boxRadius.push_back(0.5*boxRadius[j-1]);
			boxHomogRadius.push_back(pow(0.5,K->alpha)*boxHomogRadius[j-1]);
			boxLogHomogRadius.push_back(boxLogHomogRadius[j-1]-K->alpha*log(2));

			std::vector<FMM2DBox> level_vb;
			bool stop_refining = true;
			for (int k=0; k<nBoxesPerLevel[j-1]; ++k) { // no. of boxes in parent level
				/////////////////////////////////////////////////////////////////////////////////////////////
				// check if there are atleast MIN_CHARGES_PER_BOX charges inside the box to make it have children
				int s = 0;
				//	check for the charges present in parent
				if (tree[j-1][k].exists) { // if the parent is non-existent it's meaningless to have  children for such a box. so avoid checking
					for (int i=0; i<tree[j-1][k].charge_indices.size(); ++i) {
						
						if (charge_database[tree[j-1][k].charge_indices[i]].x <= (tree[j-1][k].center.x + boxRadius[j-1]) && charge_database[tree[j-1][k].charge_indices[i]].x > (tree[j-1][k].center.x - boxRadius[j-1]) && charge_database[tree[j-1][k].charge_indices[i]].y <= (tree[j-1][k].center.y + boxRadius[j-1]) && charge_database[tree[j-1][k].charge_indices[i]].y > (tree[j-1][k].center.y - boxRadius[j-1])) {
							++s;
							if(s == MIN_CHARGES_PER_BOX) {
								break;
							}
						}
					}
					
				}
				// if a box can have children it has to be 4
				
				for (int l=0; l<4; ++l) {
					FMM2DBox box;
					if (l==0) {
						box.center.x = tree[j-1][k].center.x - boxRadius[j];
						box.center.y = tree[j-1][k].center.y - boxRadius[j];
					}
					else if (l==1) {
						box.center.x = tree[j-1][k].center.x + boxRadius[j];
						box.center.y = tree[j-1][k].center.y - boxRadius[j];
					}
					else if (l==2) {
						box.center.x = tree[j-1][k].center.x + boxRadius[j];
						box.center.y = tree[j-1][k].center.y + boxRadius[j];
					}
					else {
						box.center.x = tree[j-1][k].center.x - boxRadius[j];
						box.center.y = tree[j-1][k].center.y + boxRadius[j];
					}
					// writing childrenNumbers, boxNumber, parentNumber irrespective of whether they exist or not because it helps later in forming the 4 lists					
					tree[j-1][k].childrenNumbers[l]	=	4*k+l; 
				 	box.boxNumber		=	4*k+l;
					box.parentNumber	=	k;

					if(s == MIN_CHARGES_PER_BOX) { // can have 4 children. and these
						
						box.exists		=	true;
						stop_refining = false;
						// parent cannot have children if there are not atleast enough charges in it
						// in this case parent can have children, so create the child boxes and add them to the tree
						
						// distribution of charges among the children
						// if charge is on right and top boundary of box it belongs to that box
						//cout << "center x: " << box.center.x << "	center y: " << box.center.y << "			radius: " << boxRadius[j] << endl;
						for (int i=0; i<tree[j-1][k].charge_indices.size(); ++i) {
							
							
							if (charge_database[tree[j-1][k].charge_indices[i]].x <= (box.center.x + boxRadius[j]) && charge_database[tree[j-1][k].charge_indices[i]].x > (box.center.x - boxRadius[j]) && charge_database[tree[j-1][k].charge_indices[i]].y <= (box.center.y + boxRadius[j]) && charge_database[tree[j-1][k].charge_indices[i]].y > (box.center.y - boxRadius[j])) {
								box.charge_indices.push_back(tree[j-1][k].charge_indices[i]);	
							
							}
						}
						//level.push_back(box); // putting a box into tree only if the parent box has atleast MIN_CHARGES_PER_BOX charges because a box is a box only if it's parent has atleast MIN_CHARGES_PER_BOX charges
					}	
					else {
						box.exists		=	false;
						if (l == 0) { // writing into childless boxes only once
							level_box lb;
							lb.level = j-1;
							lb.box = k;
							if (tree[j-1][k].exists) {
								childless_boxes.push_back(lb);
							}
						}
						// meaningless to have parentNumber to a non-existing child
					}
					level_vb.push_back(box); // putting non existent boxes also into tree
				}
			}
			if (stop_refining) 
				break;
			else {
				tree.push_back(level_vb);
				++j;
			}
			
		}
		nLevels = j-1;
		cout << "nLevels: " << nLevels << endl;
		smallestBoxSize	=	boxRadius[nLevels];
		a		=	smallestBoxSize;
		N		=	rank*childless_boxes.size();
		
		std::vector<FMM2DBox> level_vb;
		for (int k=0; k<4*nBoxesPerLevel[nLevels]; ++k) {
			FMM2DBox box;
			box.exists = false;
			level_vb.push_back(box);
		}
		tree.push_back(level_vb);
	}


	//	Assigns the interactions for child0 of a box
	void assign_Child0_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k;
		int nN, nNC;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N5  |  N4  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  **  |  N3  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[3]	=	nC+1;
			tree[nL][nC].neighborNumbers[4]	=	nC+2;
			tree[nL][nC].neighborNumbers[5]	=	nC+3;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*				   ______	*/
			/*				  |		 |	*/
			/*				  |	 **  |	*/
			/*	 _____________|______|  */
			/*	|	   |	  |			*/
			/*	|  I15 |  N0  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I0  |  I1  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[0];
			nNC	=	4*nN;
			if (nN != -1 ) {
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[2];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[1];
					tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[3];				
				}
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 ______			  		*/
			/*	|	   |	  			*/
			/*	|  **  |				*/
			/*	|______|______			*/
			/*	|	   |	  |			*/
			/*	|  N1  |  N2  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I2  |  I3  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			nNC	=	4*nN;
			if (nN != -1 ) {
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[1]	=	tree[j][nN].childrenNumbers[3];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[1];
				}
			}
		}

		//	Assign children of parent's second neighbor
		{
			/************************************/
			/*	 ______			  				*/
			/*	|	   |	  					*/
			/*	|  **  |						*/
			/*	|______|	   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  I5  |  O8  |	*/
			/*				  |______|______|	*/
			/*				  |	     |	    |	*/
			/*				  |  I4  |  O7  |	*/
			/*				  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[2];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[7]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[8]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  I7  |  O10 |	*/
			/*	 ______		  |______|______|	*/
			/*	|	   |	  |	     |	    |	*/
			/*	|  **  |	  |  I6  |  O9  |	*/
			/*	|______|	  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[3];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[9]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[10]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[3];
			}

		}

		//	Assign children of parent's fourth neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  O13 |  O12 |	*/
			/*				  |______|______|	*/
			/*				  |	     |	    |	*/
			/*		    	  |  I8  |  O11 |	*/
			/*		    	  |______|______|	*/
			/*									*/
			/*									*/
			/*	 ______							*/
			/*  |      |						*/
			/*  |  **  |						*/
			/*  |______|						*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[4];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[11]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[12]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[13]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  O15 |  O14 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  I10 |  I9  |		*/
			/*	|______|______|		*/
			/*						*/
			/*						*/
			/*	 ______				*/
			/*  |	   |			*/
			/*	|  **  |			*/
			/*	|______|			*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[14]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[15]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/****************************/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  O17 |  O16 |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I12 |  I11 |			*/
			/*	|______|______|			*/
			/*							*/
			/*							*/
			/*				   ______	*/
			/*  			  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[6];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[16]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[17]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/****************************/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I13 |  N6  |			*/
			/*	|______|______|______	*/
			/*  |	   |	  |		 |	*/
			/*	|  I14 |  N7  |	 **  |	*/
			/*	|______|______|______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[7];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[7]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[2];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[3];				
				}
			}
		}
	}

	//	Assigns the interactions for child1 of a box
	void assign_Child1_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+1;
		int nN,nNC;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N6  |  N5  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N7  |  **  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[7]	=	nC-1;
			tree[nL][nC].neighborNumbers[5]	=	nC+1;
			tree[nL][nC].neighborNumbers[6]	=	nC+2;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/************************************/
			/*				   		  ______	*/
			/*				  	     |		|	*/
			/*				         |	**  |	*/
			/*	 _____________       |______|  	*/
			/*	|	   |	  |					*/
			/*	|  O22 |  I15 |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O23 |  I0  |					*/
			/*	|______|______|					*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[0];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[23]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[22]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 		______		  	*/
			/*		   |	  |			*/
			/*	       |  **  |			*/
			/*	 ______|______|			*/
			/*	|	   |	  |			*/
			/*	|  N0  |  N1  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I1  |  I2  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[1]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[3];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				}								
			}
		}

		//	Assign children of parent's second neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |	  			*/
			/*	|______|_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  N2  |  I5  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*	       |  I3  |  I4  |	*/
			/*		   |______|______|	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[2];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[3];				
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[1];
					tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[2];
				}
			}
		}

		//	Assign children of parent's third neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  N4  |	 I7	 |  */
			/*	 ______|______|______|	*/
			/*	|	   |	  |	     |	*/
			/*	|  **  |  N3  |  I6  |  */
			/*	|______|______|______| 	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[3];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[3]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[3];	
				if(tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[1];
					tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[2];
				}							
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  O14 |  O13 |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  I9  |  I8  |	*/
			/*		   |______|______|	*/
			/*				  			*/
			/*				  			*/
			/*	 ______					*/
			/*	|	   |				*/
			/*  |  **  |				*/
			/*  |______|				*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[4];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[13]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[14]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  O16 |  O15 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  I11 |  I10 |		*/
			/*	|______|______|		*/
			/*						*/
			/*						*/
			/*		    ______		*/
			/* 		   |	  |		*/
			/*		   |  **  |		*/
			/*		   |______|		*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[15]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[16]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/************************************/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O18 |  O17 |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O19 |  I12 |					*/
			/*	|______|______|					*/
			/*									*/
			/*									*/
			/*				   		  ______	*/
			/*  			  		 |		|	*/
			/*				  		 |	** 	|	*/
			/*				  		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[6];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[19]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[17]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[18]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/************************************/
			/*									*/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O20 |  I13 |					*/
			/*	|______|______|		  ______	*/
			/*  |	   |	  |		 |		|	*/
			/*	|  O21 |  I14 |	 	 |	**  |	*/
			/*	|______|______|		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[7];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[21]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[20]	=	tree[j][nN].childrenNumbers[3];				
			}
		}
	}

	//	Assigns the interactions for child2 of a box
	void assign_Child2_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+2;
		int nN,nNC;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N7  |  **  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N0  |  N1  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[0]	=	nC-2;
			tree[nL][nC].neighborNumbers[1]	=	nC-1;
			tree[nL][nC].neighborNumbers[7]	=	nC+1;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/************************************/
			/*				   		  ______	*/
			/*				  	     |		|	*/
			/*				         |	**  |	*/
			/*				         |______|  	*/
			/*									*/
			/*									*/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O23 |  I0  |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O0  |  O1  |					*/
			/*	|______|______|					*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[0];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[1]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[23]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 		______		  	*/
			/*		   |	  |			*/
			/*	       |  **  |			*/
			/*	 	   |______|			*/
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I1  |  I2  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O2  |  O3  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[3]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's second neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |	  			*/
			/*	|______|				*/
			/*							*/
			/*							*/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  I3  |  I4  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*	       |  O4  |  O5  |	*/
			/*		   |______|______|	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[2];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[5]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/****************************/
			/*	 ____________________	*/
			/*	|	   |	  |	     |	*/
			/*	|  **  |  N3  |	 I6	 |  */
			/*	|______|______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  N2  |  I5  |  */
			/*		   |______|______| 	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[3];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[3]	=	tree[j][nN].childrenNumbers[3];		
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[1];
					tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[2];
				}
						
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  I9  |  I8  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  N4  |  I7  |	*/
			/*	 ______|______|______|	*/
			/*	|	   |	  			*/
			/*	|  **  |	  			*/
			/*	|______|	  			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[4];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[1];
					tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[2];
					tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[3];				
				}
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  I11 |  I10 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N6  |  N5  |		*/
			/*	|______|______|		*/
			/*		   |	  |		*/
			/*		   |  **  |		*/
			/*		   |______|		*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[5]	=	tree[j][nN].childrenNumbers[1];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[2];
					tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[3];				
				}
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/************************************/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O19 |  I12 |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O20 |  I13 |					*/
			/*	|______|______|		  ______	*/
			/*  			  		 |		|	*/
			/*				  		 |	** 	|	*/
			/*				  		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[6];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[20]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[19]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/************************************/
			/*									*/
			/*	 _____________		  ______	*/
			/*	|	   |	  |		 |	    |	*/
			/*	|  O21 |  I14 |		 |	**	|	*/
			/*	|______|______|		 |______|	*/
			/*  |	   |	  |		 			*/
			/*	|  O22 |  I15 |	 	 			*/
			/*	|______|______|		 			*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[7];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[22]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[21]	=	tree[j][nN].childrenNumbers[3];				
			}
		}
	}

	//	Assigns the interactions for child3 of a box
	void assign_Child3_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+3;
		int nN,nNC;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  **  |  N3  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N1  |  N2  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[1]	=	nC-3;
			tree[nL][nC].neighborNumbers[2]	=	nC-2;
			tree[nL][nC].neighborNumbers[3]	=	nC-1;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*				   ______	*/
			/*				  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|  */
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I0  |  I1  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O1  |  O2  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[0];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[1]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |				*/
			/*	|______|				*/
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I2  |  I3  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O3  |  O4  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[3]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[4]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's second neighbor
		{
			/************************************/
			/*	 ______		  					*/
			/*	|	   |						*/
			/*	|  **  |	  					*/
			/*	|______|						*/
			/*									*/
			/*									*/
			/*				   _____________	*/
			/*		   		  |	     |	    |	*/
			/*		   		  |  I4  |  O7  |	*/
			/*		   		  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*	       		  |  O5  |  O6  |	*/
			/*		   		  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[2];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[6]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[7]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/************************************/
			/*	 ______		   _____________	*/
			/*	|	   |	  |	     |		|	*/
			/*	|  **  |      |	 I6	 |  O9	|	*/
			/*	|______|	  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I5  |  O8  |  	*/
			/*		   		  |______|______| 	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[3];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[8]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[9]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I8  |  O11 |	*/
			/*		   		  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I7  |  O10 |	*/
			/*	 ______	      |______|______|	*/
			/*	|	   |	  					*/
			/*	|  **  |	  					*/
			/*	|______|	  					*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[4];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[10]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[11]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  I10 |  I9  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N5  |  N4  |		*/
			/*	|______|______|		*/
			/*	|	   |			*/
			/*	|  **  |			*/
			/*	|______|			*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[1];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[2];
					tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[3];				
				}
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/****************************/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I12 |  I11 |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I13 |  N6  |			*/
			/*	|______|______|______	*/
			/*  			  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[6];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[1];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[2];
					tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[3];				
				}
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/****************************/
			/*							*/
			/*	 ____________________	*/
			/*	|	   |	  |		 |	*/
			/*	|  I14 |  N7  |	 **	 |	*/
			/*	|______|______|______|	*/
			/*  |	   |	  |		 	*/
			/*	|  I15 |  N0  |	 	 	*/
			/*	|______|______|		 	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[7];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[7]	=	tree[j][nN].childrenNumbers[2];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[3];	
				}			
			}
		}
	}

	//	Assigns the interactions for the children of a box
	void assign_Box_Interactions(int j, int k) {
		assign_Child0_Interaction(j,k);
		assign_Child1_Interaction(j,k);
		assign_Child2_Interaction(j,k);
		assign_Child3_Interaction(j,k);
	}

	//	Assigns the interactions for the children all boxes at a given level
	void assign_Level_Interactions(int j) {
		//cout << endl << "tree[j].size: " << tree[j].size() << endl;
		//#pragma omp parallel for
		for (int k=0; k<tree[j].size(); ++k) {
			//cout << k << endl;
			if (tree[j+1][4*k].exists) { // do this only if the box is a parent
				//cout << j+1 << "," << 4*k << "level box" << endl;
				assign_Box_Interactions(j,tree[j][k].boxNumber);
			}
		}
	}

	//	Assigns the interactions for the children of all boxes in the tree
	//	Assigns colleagues(neighbors) and list2 (inner and outer numbers) needed for M2L(same size)
	void assign_list2_Neighbor_Interactions() {
		for (int j=0; j<nLevels; ++j) {
			assign_Level_Interactions(j);
		}
	}


	//	Assigns list1 for childless boxes	
	void assign_list1_list3_Interactions() {
		for (int j=0; j<childless_boxes.size(); ++j) {
			//childless_boxes[j].level = l;
			//childless_boxes[j].box = b;
			assign_list1_list3_box_Interactions(childless_boxes[j]);
		}
	}





	void check2 () {
		cout << endl << "tree[5][512].exists: " << tree[5][512].exists << endl;
		cout << endl << "tree[4][128].childrenNumbers[0]: " << tree[4][128].childrenNumbers[0] << endl;
	}


	void check () {
		for (int i=0; i<8; ++i) {
			cout << tree[1][1].neighborNumbers[i] << ",";
		}
		//cout << endl;
		for (int j=0; j<childless_boxes.size(); ++j) {
			cout << endl <<  "j: " << childless_boxes[j].level << "	k: " << childless_boxes[j].box << endl;
			cout << "list1: " << endl;
			for (int i=0; i<tree[childless_boxes[j].level][childless_boxes[j].box].list1.size() ; ++i) {
				cout << "j: " << tree[childless_boxes[j].level][childless_boxes[j].box].list1[i].level << "	k: " << tree[childless_boxes[j].level][childless_boxes[j].box].list1[i].box << endl;
			}

			cout << "list3: " << endl;
			for (int i=0; i<tree[childless_boxes[j].level][childless_boxes[j].box].list3.size() ; ++i) {
				cout << "j: " << tree[childless_boxes[j].level][childless_boxes[j].box].list3[i].level << "	k: " << tree[childless_boxes[j].level][childless_boxes[j].box].list3[i].box << endl;
			}
		}

		/*for (int j=0; j<childless_boxes.size(); ++j) {
			cout << "j: " << childless_boxes[j].level << "	k: " << childless_boxes[j].box << endl;

		}
		for (int j=0; j<=nLevels; ++j) {
			cout << "j: " << j << endl;
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				if (tree[j][k].exists) {
					cout << k << ","; 
				}
			}
			cout << endl;
		}

		for (int j=0; j<=nLevels; ++j) {
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				if (tree[j][k].childrenNumbers[0] != -1) {
					for(int c=0 ;c<4; ++c) {
						cout << "j,k:  " << j+1 << "," << 4*k+c << endl;
						cout << "neighborNumbers" << endl; 
						for (int i=0; i<8; ++i) {
							cout << tree[j+1][4*k+c].neighborNumbers[i] << ",";
						}
						cout << endl;
	
						cout << "innerNumbers" << endl; 
						for (int i=0; i<16; ++i) {
							cout << tree[j+1][4*k+c].innerNumbers[i] << ",";
						}
						cout << endl;
			
						cout << "outerNumbers" << endl; 
						for (int i=0; i<24; ++i) {
							cout << tree[j+1][4*k+c].outerNumbers[i] << ",";
						}
						cout << endl << endl;
					}
				}

				cout << "center x: " << tree[j][k].center.x  << "	" << "center y: " << tree[j][k].center.y << endl;
				cout << "box exists: " << tree[j][k].exists << endl;
				cout << "charges" << endl;
				for (int c=0; c<tree[j][k].charge_indices.size(); ++c) {
					cout << charge_database[tree[j][k].charge_indices[c]].x << "	" <<  charge_database[tree[j][k].charge_indices[c]].y << endl;
				}
				cout << endl;
			}
			cout << endl;
		}*/
	}


	//	LIST 1
	void assign_list1_list3_box_Interactions(const level_box lb) { // for boxes which donot have children
		level_box temp;
		
		//	if box is childless it also is a member of list 1
		tree[lb.level][lb.box].list1.push_back(lb);
		level_box prev_add, neigh0_add;
		prev_add.level = -1;
		prev_add.box = 0;
		neigh0_add.level = -1;
		neigh0_add.box = 0;
		int j,k,cN;
		
		//	NEIGHBOR 0,2,4,6 CORNERS
		//	NEIGHBOR 1,3,5,7 SIDE BY SIDE

		//	NEIGHBOR 0
		
		k = tree[lb.level][lb.box].neighborNumbers[0]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { // k=-1 means that box is not in the computational domain
			cN = k;
		if (tree[j][k].exists) {
			
			while (tree[j+1][4*cN].exists) {
				
				//	list 3
				
					temp.level = j+1;
					temp.box = 4*cN;
					tree[lb.level][lb.box].list3.push_back(temp);
					tree[temp.level][temp.box].list4.push_back(lb);	

					temp.level = j+1;
					temp.box = 4*cN+1;
					tree[lb.level][lb.box].list3.push_back(temp);
					tree[temp.level][temp.box].list4.push_back(lb);	

					temp.level = j+1;
					temp.box = 4*cN+3;
					tree[lb.level][lb.box].list3.push_back(temp);
					tree[temp.level][temp.box].list4.push_back(lb);	
					
					j = j+1;
					cN = 4*cN+2;
				
				/////////////////////////////////////////////
			}
			
			temp.level = j;
			temp.box = cN;
			tree[lb.level][lb.box].list1.push_back(temp);
			/*//	list 3
			k = tree[lb.level][lb.box].neighborNumbers[0]; // neighbor is in same level
			j = lb.level;
			cN = 4*k+2;
			if (tree[j+1][cN].exists) {
				temp.level = j;
				temp.box = cN/4-2;
				tree[lb.level][lb.box].list3.push_back(temp);
			
				temp.level = j;
				temp.box = cN/4-1;
				tree[lb.level][lb.box].list3.push_back(temp);
		
				temp.level = j;
				temp.box = cN/4+1;
				tree[lb.level][lb.box].list3.push_back(temp);
			}*/
			/////////////////////////////////////////////

		}
		else {
			// immendiate next valid parent // goto higher levels
			while (tree[j][k].exists == false) {		
				j = j-1;
				k = k/4;
			}
			temp.level = j;
			temp.box = k;
			tree[lb.level][lb.box].list1.push_back(temp);
			prev_add = temp;
			neigh0_add = temp;
			
		}
		}

		
		//	NEIGHBOR 1
		k = tree[lb.level][lb.box].neighborNumbers[1]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { 
			if (tree[j][k].exists) {
				list1_neighbor1(lb.level,lb.box,j,k);
			}
		}


		
		//	NEIGHBOR 2
		
		k = tree[lb.level][lb.box].neighborNumbers[2]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { 
			cN = k;
		if (tree[j][k].exists) {
			
			while (tree[j+1][4*cN].exists) {
				
				//	list 3
				
					temp.level = j+1;
					temp.box = 4*cN;
					tree[lb.level][lb.box].list3.push_back(temp);
					tree[temp.level][temp.box].list4.push_back(lb);	
					
					temp.level = j+1;
					temp.box = 4*cN+1;
					tree[lb.level][lb.box].list3.push_back(temp);
					tree[temp.level][temp.box].list4.push_back(lb);	
					
					temp.level = j+1;
					temp.box = 4*cN+2;
					tree[lb.level][lb.box].list3.push_back(temp);
					tree[temp.level][temp.box].list4.push_back(lb);	
					
					
				
				j = j+1;
				cN = 4*cN+3;
			}
				/////////////////////////////////////////////
			
			temp.level = j;
			temp.box = cN;
			tree[lb.level][lb.box].list1.push_back(temp);
			
			/*//	list 3
			k = tree[lb.level][lb.box].neighborNumbers[2]; // neighbor is in same level
			j = lb.level;
			cN = 4*k+3;
			if (tree[j+1][cN].exists) {
				temp.level = j;
				temp.box = cN/4-3;
				tree[lb.level][lb.box].list3.push_back(temp);
			
				temp.level = j;
				temp.box = cN/4-2;
				tree[lb.level][lb.box].list3.push_back(temp);
	
				temp.level = j;
				temp.box = cN/4-1;
				tree[lb.level][lb.box].list3.push_back(temp);
			}*/
			/////////////////////////////////////////////

		}
		else {
			// immendiate next valid parent // goto higher levels
			while (tree[j][k].exists == false) {		
				j = j-1;
				k = k/4;
			}
			temp.level = j;
			temp.box = k;
			if (prev_add.level == j && prev_add.box == k) {
				
			}
			else {
				tree[lb.level][lb.box].list1.push_back(temp);
				
			}
			prev_add = temp;
		}
		}


		

		//	NEIGHBOR 3
		k = tree[lb.level][lb.box].neighborNumbers[3]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { 
			if (tree[j][k].exists) {
				list1_neighbor3(lb.level,lb.box,j,k);
			}
		}



		//	NEIGHBOR 4
		
		k = tree[lb.level][lb.box].neighborNumbers[4]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { 
			cN = k;
		if (tree[j][k].exists) {
			

			while (tree[j+1][4*cN].exists) {
				
				
				//	list 3
				
					temp.level = j+1;
					temp.box = 4*cN+1;
					tree[lb.level][lb.box].list3.push_back(temp);
					tree[temp.level][temp.box].list4.push_back(lb);	
					
					temp.level = j+1;
					temp.box = 4*cN+2;
					tree[lb.level][lb.box].list3.push_back(temp);
					tree[temp.level][temp.box].list4.push_back(lb);	
					
					temp.level = j+1;
					temp.box = 4*cN+3;
					tree[lb.level][lb.box].list3.push_back(temp);
					tree[temp.level][temp.box].list4.push_back(lb);	
					
					
				j = j+1;
				cN = 4*cN;
				/////////////////////////////////////////////
	
			}
			temp.level = j;
			temp.box = cN;
			tree[lb.level][lb.box].list1.push_back(temp);
			
			/*//	list 3
			k = tree[lb.level][lb.box].neighborNumbers[4]; // neighbor is in same level
			j = lb.level;
			cN = 4*k;
			if (tree[j+1][cN].exists) {
				temp.level = j;
				temp.box = cN/4+1;
				tree[lb.level][lb.box].list3.push_back(temp);
			
				temp.level = j;
				temp.box = cN/4+2;
				tree[lb.level][lb.box].list3.push_back(temp);
		
				temp.level = j;
				temp.box = cN/4+3;
				tree[lb.level][lb.box].list3.push_back(temp);
			}*/
			/////////////////////////////////////////////

		}
		else {
			// immendiate next valid parent // goto higher levels
			while (tree[j][k].exists == false) {		
				j = j-1;
				k = k/4;
			}
			temp.level = j;
			temp.box = k;
			if (prev_add.level == j && prev_add.box == k) {
				
			}
			else {
				tree[lb.level][lb.box].list1.push_back(temp);
				
			}
			prev_add = temp;
		}
		}


		


		//	NEIGHBOR 5
		k = tree[lb.level][lb.box].neighborNumbers[5]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { 
			if (tree[j][k].exists) {
				
				list1_neighbor5(lb.level,lb.box,j,k);
			}
		}




		//	NEIGHBOR 6
		
		k = tree[lb.level][lb.box].neighborNumbers[6]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { 
			cN = k;
		if (tree[j][k].exists) {
			
			while (tree[j+1][4*cN].exists) {
				
							
				//	list 3
				
					temp.level = j+1;
					temp.box = 4*cN;
					tree[lb.level][lb.box].list3.push_back(temp);
					tree[temp.level][temp.box].list4.push_back(lb);	
					
					temp.level = j+1;
					temp.box = 4*cN+2;
					tree[lb.level][lb.box].list3.push_back(temp);
					tree[temp.level][temp.box].list4.push_back(lb);	
					
					temp.level = j+1;
					temp.box = 4*cN+3;
					tree[lb.level][lb.box].list3.push_back(temp);
					tree[temp.level][temp.box].list4.push_back(lb);	
					
					j = j+1;
					cN = 4*cN+1;
					
				/////////////////////////////////////////////
			}
			temp.level = j;
			
			temp.box = cN;
			tree[lb.level][lb.box].list1.push_back(temp);
			
			/*//	list 3
			k = tree[lb.level][lb.box].neighborNumbers[6]; // neighbor is in same level
			j = lb.level;
			cN = 4*k+1;
			if (tree[j+1][cN].exists) {
				temp.level = j;
				temp.box = cN/4-1;
				tree[lb.level][lb.box].list3.push_back(temp);
		
				temp.level = j;
				temp.box = cN/4+1;
				tree[lb.level][lb.box].list3.push_back(temp);
		
				temp.level = j;
				temp.box = cN/4+2;
				tree[lb.level][lb.box].list3.push_back(temp);
			}*/
			/////////////////////////////////////////////

		}
		else {
			// immendiate next valid parent // goto higher levels
			while (tree[j][k].exists == false) {		
				j = j-1;
				k = k/4;
			}
			temp.level = j;
			temp.box = k;
			if ((prev_add.level == j && prev_add.box == k) || (neigh0_add.level == j && neigh0_add.box == k)) {
				
			}
			else {
				tree[lb.level][lb.box].list1.push_back(temp);
				
			}
			prev_add = temp;
		}
		}


		
		
		//	NEIGHBOR 7
		k = tree[lb.level][lb.box].neighborNumbers[7]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { 
			if (tree[j][k].exists) {
				list1_neighbor7(lb.level,lb.box,j,k);
			}
		}
	}

	

	void list1_neighbor1(int Lj, int Lk, int Nj, int Nk) {
		level_box temp;
		if (tree[Nj+1][4*Nk].exists) {
			level_box lb;
			lb.level = Lj;
			lb.box = Lk;
			// list 3
			temp.level = Nj+1;
			temp.box = 4*Nk;
			tree[Lj][Lk].list3.push_back(temp);
			tree[temp.level][temp.box].list4.push_back(lb);
		
			temp.box = 4*Nk+1;
			tree[Lj][Lk].list3.push_back(temp);
			tree[temp.level][temp.box].list4.push_back(lb);
			//////////////////////////////////

			list1_neighbor1(Lj,Lk,Nj+1,4*Nk+2);
			list1_neighbor1(Lj,Lk,Nj+1,4*Nk+3);

		}
		else {
			temp.level = Nj;
			temp.box = Nk;
			tree[Lj][Lk].list1.push_back(temp);

			/*// list 3	//siblings of list1 		
			temp.level = Nj;
			if (Nk%4 == 2) {
				temp.box = Nk-1;
				tree[Lj][Lk].list3.push_back(temp);
			}
			else if (Nk%4 == 3) {
				temp.box = Nk-3;
				tree[Lj][Lk].list3.push_back(temp);
			}*/
			//////////////////////////////////
		}
	}



	void list1_neighbor3(int Lj, int Lk, int Nj, int Nk) {
		level_box temp;
		if (tree[Nj+1][4*Nk].exists) {
			level_box lb;
			lb.level = Lj;
			lb.box = Lk;
			// list 3
			temp.level = Nj+1;
			temp.box = 4*Nk+1;
			tree[Lj][Lk].list3.push_back(temp);
			tree[temp.level][temp.box].list4.push_back(lb);
		
			temp.box = 4*Nk+2;
			tree[Lj][Lk].list3.push_back(temp);
			tree[temp.level][temp.box].list4.push_back(lb);
		
			//////////////////////////////////			
			
			list1_neighbor3(Lj,Lk,Nj+1,4*Nk+0);
			list1_neighbor3(Lj,Lk,Nj+1,4*Nk+3);

		}
		else {
			temp.level = Nj;
			temp.box = Nk;
			tree[Lj][Lk].list1.push_back(temp);

			/*// list 3	//siblings of list1 		
			temp.level = Nj;
			if (Nk%4 == 0) {
				temp.box = Nk+1;
				tree[Lj][Lk].list3.push_back(temp);
			}
			else if (Nk%4 == 3) {
				temp.box = Nk-1;
				tree[Lj][Lk].list3.push_back(temp);
			}*/
			//////////////////////////////////
		}
	}




	void list1_neighbor5(int Lj, int Lk, int Nj, int Nk) {
		level_box temp;
		if (tree[Nj+1][4*Nk].exists) {
			level_box lb;
			lb.level = Lj;
			lb.box = Lk;
			// list 3
			temp.level = Nj+1;
			temp.box = 4*Nk+2;
			tree[Lj][Lk].list3.push_back(temp);
			tree[temp.level][temp.box].list4.push_back(lb);
				
			temp.box = 4*Nk+3;
			tree[Lj][Lk].list3.push_back(temp);
			tree[temp.level][temp.box].list4.push_back(lb);
		
			//////////////////////////////////

			list1_neighbor5(Lj,Lk,Nj+1,4*Nk);
			list1_neighbor5(Lj,Lk,Nj+1,4*Nk+1);

					
		}
		else {
			temp.level = Nj;
			temp.box = Nk;
			tree[Lj][Lk].list1.push_back(temp);

			/*// list 3	//siblings of list1 		
			temp.level = Nj;
			if (Nk%4 == 0) {
				temp.box = Nk+3;
				tree[Lj][Lk].list3.push_back(temp);
			}
			else if (Nk%4 == 1) {
				temp.box = Nk+1;
				tree[Lj][Lk].list3.push_back(temp);
			}*/
			//////////////////////////////////
		}
	}




	void list1_neighbor7(int Lj, int Lk, int Nj, int Nk) {
		level_box temp;
		if (tree[Nj+1][4*Nk].exists) {
			level_box lb;
			lb.level = Lj;
			lb.box = Lk;
			// list 3
			temp.level = Nj+1;
			temp.box = 4*Nk;
			tree[Lj][Lk].list3.push_back(temp);
			tree[temp.level][temp.box].list4.push_back(lb);
				
			temp.box = 4*Nk+3;
			tree[Lj][Lk].list3.push_back(temp);
			tree[temp.level][temp.box].list4.push_back(lb);
		
			//////////////////////////////////	

			list1_neighbor7(Lj,Lk,Nj+1,4*Nk+1);
			list1_neighbor7(Lj,Lk,Nj+1,4*Nk+2);

			
		}
		else {
			temp.level = Nj;
			temp.box = Nk;
			tree[Lj][Lk].list1.push_back(temp);

			/*// list 3	//siblings of list1 		
			temp.level = Nj;
			if (Nk%4 == 1) {
				temp.box = Nk-1;
				tree[Lj][Lk].list3.push_back(temp);
			}
			else if (Nk%4 == 2) {
				temp.box = Nk+1;
				tree[Lj][Lk].list3.push_back(temp);
			}*/
			//////////////////////////////////
		}
	}




	//	Obtain List 4 Interactions
	void assign_list4_Interations() {
		level_box temp;
		for (int j=2; j<=nLevels; ++j) { // list 4 doesn't exist for boxes in level < 2
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
		/*for (int j=4; j<=4; ++j) { // list 4 doesn't exist for boxes in level < 2
			for (int k=129; k<=129; ++k) {*/
				if (tree[j][k].exists) { // list 4 is found only for existing boxes
					int pb = k;
					for (int pl=j-1; pl>=1; --pl) { // starting with parent
						pb = pb/4;
						for (int n=0; n<8; ++n) { // check for neighbors
							/*if (k%4 == 0) { //child 0
								if (n==0 || n==1 || n==7) {
									continue;
								}
							}
							else if (k%4 == 1) { // child 1
								if (n==1 || n==2 || n==3) {
									continue;
								}
							}
							else if (k%4 == 2) { // child 2
								if (n==3 || n==4 || n==5) {
									continue;
								}
							}
							else { // child 3
								if (n==5 || n==6 || n==7) {
									continue;
								}
 							}*/

							
							int nN = tree[pl][pb].neighborNumbers[n];
							if (nN != -1) {
								if (tree[pl][nN].exists && tree[pl+1][4*nN].exists == false) { // if they don't exist they will be checked for at next higher level
									temp.level = pl;
									temp.box = nN;
									int cnN = tree[j][k].neighborNumbers[n];
									//cout << "j: " << j << " k: " << k << " n: " << n <<  " cnN: " << cnN << " nN: " << nN << " pow((double)4,double(pl-j)): " << pow(4,j-pl) << " cnN / pow(4,j-pl): " << cnN / pow(4,j-pl) << endl;
									if (int(cnN / pow((double)4,(double)(j-pl))) == nN) {
										//cout << "yes" << endl;
									}
									else
										tree[j][k].list4.push_back(temp);
								}
							}
						}
					}
				}
			}
		}
	}




	void check3 () {
		/*cout << endl;
		for (int n=0; n<8; ++n) {
			cout << tree[4][129].neighborNumbers[n] << "," ;
		}*/
		cout << endl;
		for (int j=2; j<=nLevels; ++j) { // list 4 doesn't exist for boxes in level < 2
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
		/*for (int j=4; j<=4; ++j) { // list 4 doesn't exist for boxes in level < 2
			for (int k=129; k<=129; ++k) {*/
				if (tree[j][k].exists) { // list 4 is found only for existing boxes
					cout << endl <<  "j: " << j << "	k: " << k << endl;
					cout << "list4: " << endl;
					for (int i=0; i<tree[j][k].list4.size() ; ++i) {
						cout << "j: " << tree[j][k].list4[i].level << "	k: " << tree[j][k].list4[i].box << endl;
					}
				}	
			}
		}
	}


	//	Obtain the desired matrix
	void obtain_Desired_Operator(std::vector<pts2D>& shiftedChebNodes, Eigen::MatrixXd& T) {
		T	=	Eigen::MatrixXd(rank,rank);
		for (int i=0; i<rank; ++i) {
			for (int j=0; j<rank; ++j) {
				T(i,j)	=	K->getInteraction(standardChebNodes[i], shiftedChebNodes[j], a);
			}
		}
	}

	void obtain_Desired_Leaf_Operator(std::vector<pts2D>& shiftedChebNodes, Eigen::MatrixXd& T) {
		T	=	Eigen::MatrixXd(rank,rank);
		for (int i=0; i<rank; ++i) {
			for (int j=0; j<rank; ++j) {
				T(i,j)	=	K->getInteraction(leafChebNodes[i], shiftedChebNodes[j], a);
			}
		}
	}

	//	Assemble FMM Operators
	void assemble_Operators_FMM() {
		std::vector<pts2D> shiftedChebNodes;
		//	Assemble Outer Interactions
		for (int l=0; l<6; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(l-3,-3);
			obtain_Desired_Operator(shiftedChebNodes, M2LOuter[l]);
		}
		for (int l=0; l<6; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(3,l-3);
			obtain_Desired_Operator(shiftedChebNodes, M2LOuter[l+6]);
		}
		for (int l=0; l<6; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(3-l,3);
			obtain_Desired_Operator(shiftedChebNodes, M2LOuter[l+12]);
		}
		for (int l=0; l<6; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(-3,3-l);
			obtain_Desired_Operator(shiftedChebNodes, M2LOuter[l+18]);
		}
		//	Assemble Inner Interactions
		for (int l=0; l<4; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(l-2,-2);
			obtain_Desired_Operator(shiftedChebNodes, M2LInner[l]);
		}
		for (int l=0; l<4; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(2,l-2);
			obtain_Desired_Operator(shiftedChebNodes, M2LInner[l+4]);
		}
		for (int l=0; l<4; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(2-l,2);
			obtain_Desired_Operator(shiftedChebNodes, M2LInner[l+8]);
		}
		for (int l=0; l<4; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(-2,2-l);
			obtain_Desired_Operator(shiftedChebNodes, M2LInner[l+12]);
		}

/*
		//	Assigning Leaf Interactions
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	boxRadius[nLevels]*standardChebNodes[k].x;
			temp.y	=	boxRadius[nLevels]*standardChebNodes[k].y;
			leafChebNodes.push_back(temp);
		}
		//	Assemble Neighbor Interactions
		double neighborDistance	=	2.0*boxRadius[nLevels];
		{
			shiftedChebNodes	=	shift_Leaf_Cheb_Nodes(-neighborDistance,-neighborDistance);
			obtain_Desired_Leaf_Operator(shiftedChebNodes, neighborInteraction[0]);
		}
		{
			shiftedChebNodes	=	shift_Leaf_Cheb_Nodes(0,-neighborDistance);
			obtain_Desired_Leaf_Operator(shiftedChebNodes, neighborInteraction[1]);
		}
		{
			shiftedChebNodes	=	shift_Leaf_Cheb_Nodes(neighborDistance,-neighborDistance);
			obtain_Desired_Leaf_Operator(shiftedChebNodes, neighborInteraction[2]);
		}
		{
			shiftedChebNodes	=	shift_Leaf_Cheb_Nodes(neighborDistance,0);
			obtain_Desired_Leaf_Operator(shiftedChebNodes, neighborInteraction[3]);
		}
		{
			shiftedChebNodes	=	shift_Leaf_Cheb_Nodes(neighborDistance,neighborDistance);
			obtain_Desired_Leaf_Operator(shiftedChebNodes, neighborInteraction[4]);
		}
		{
			shiftedChebNodes	=	shift_Leaf_Cheb_Nodes(0,neighborDistance);
			obtain_Desired_Leaf_Operator(shiftedChebNodes, neighborInteraction[5]);
		}
		{
			shiftedChebNodes	=	shift_Leaf_Cheb_Nodes(-neighborDistance,neighborDistance);
			obtain_Desired_Leaf_Operator(shiftedChebNodes, neighborInteraction[6]);
		}
		{
			shiftedChebNodes	=	shift_Leaf_Cheb_Nodes(-neighborDistance,0);
			obtain_Desired_Leaf_Operator(shiftedChebNodes, neighborInteraction[7]);
		}
		//	Assemble Self Interactions
		{
			obtain_Desired_Leaf_Operator(leafChebNodes, selfInteraction);
		}
*/
	}


	
/*	void assign_Center_Location() {
		int J, K;
		tree[0][0].center.x	=	0.0;
		tree[0][0].center.y	=	0.0;
		for (int j=0; j<nLevels; ++j) {
			J	=	j+1;
			double shift	=	0.5*boxRadius[j];
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				K	=	4*k;
				tree[J][K].center.x		=	tree[j][k].center.x-shift;
				tree[J][K+1].center.x	=	tree[j][k].center.x+shift;
				tree[J][K+2].center.x	=	tree[j][k].center.x+shift;
				tree[J][K+3].center.x	=	tree[j][k].center.x-shift;

				tree[J][K].center.y		=	tree[j][k].center.y-shift;
				tree[J][K+1].center.y	=	tree[j][k].center.y-shift;
				tree[J][K+2].center.y	=	tree[j][k].center.y+shift;
				tree[J][K+3].center.y	=	tree[j][k].center.y+shift;
			}
		}
		#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[nLevels]; ++k) {
			tree[nLevels][k].chebNodes	=	shift_Leaf_Cheb_Nodes(tree[nLevels][k].center.x, tree[nLevels][k].center.y);
		}
	}

	void assign_Leaf_Charges() {
		#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[nLevels]; ++k) {
			tree[nLevels][k].multipoles	=	0.5*(Eigen::VectorXd::Ones(rank)+Eigen::VectorXd::Random(rank));
		}
	}
*/



	//	assign_Leaf_ChebNodes
	void assign_Leaf_ChebNodes() {
		for (int j=0; j<childless_boxes.size(); ++j) {
			level_box lb = childless_boxes[j];
			tree[lb.level][lb.box].chebNodes	=	shift_scale_Cheb_Nodes(tree[lb.level][lb.box].center.x,tree[lb.level][lb.box].center.y,boxRadius[lb.level]);
		}
	}




	// evaluate multipoles
	void evaluate_multipoles() {
		for (int j=0; j<childless_boxes.size(); ++j) { // leafs
			level_box lb = childless_boxes[j];
			tree[lb.level][lb.box].multipoles = Eigen::VectorXd::Zero(rank);
			for (int m=0; m<rank; ++m) { // multipoles
				for (int i=0; i<tree[lb.level][lb.box].charge_indices.size(); ++i) { // charges in leaf
					tree[lb.level][lb.box].multipoles[m] += charge_database[tree[lb.level][lb.box].charge_indices[i]].q * get_S(tree[lb.level][lb.box].chebNodes[m].x, charge_database[tree[lb.level][lb.box].charge_indices[i]].x, nChebNodes) * get_S(tree[lb.level][lb.box].chebNodes[m].y, charge_database[tree[lb.level][lb.box].charge_indices[i]].y, nChebNodes);
				}
			}
		}
	}



	void evaluate_All_M2M() {
		for (int j=nLevels-1; j>1; --j) { // parent
			int J	=	j+1; // children
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) { // parent
				int K	=	4*k; // children
				if (tree[J][K].exists) {
					tree[j][k].multipoles	=	M2M[0]*tree[J][K].multipoles+M2M[1]*tree[J][K+1].multipoles+M2M[2]*tree[J][K+2].multipoles+M2M[3]*tree[J][K+3].multipoles;
				}
			}
		}
	}

	void evaluate_list2() { // list 2
		for (int j=0; j<=1; ++j) {
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				if (tree[j][k].exists) {
					tree[j][k].locals	=	Eigen::VectorXd::Zero(rank);
				}
			}
		}
		for (int j=2; j<=nLevels; ++j) {
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				if (tree[j][k].exists) {
					tree[j][k].locals	=	Eigen::VectorXd::Zero(rank);
					#ifdef HOMOG
						//	Inner well-separated clusters
						for (int l=0; l<16; ++l) {
							int nInner	=	tree[j][k].innerNumbers[l];
							if (nInner>-1) {
								tree[j][k].locals+=M2LInner[l]*tree[j][nInner].multipoles;
							}
						}
						//	Outer well-separated clusters
						for (int l=0; l<24; ++l) {
							int nOuter	=	tree[j][k].outerNumbers[l];
							if (nOuter>-1) {
								tree[j][k].locals+=M2LOuter[l]*tree[j][nOuter].multipoles;
							}
						}
						tree[j][k].locals*=boxHomogRadius[j];					
					#elif LOGHOMOG
						//	Inner well-separated clusters
						for (int l=0; l<16; ++l) {
							int nInner	=	tree[j][k].innerNumbers[l];
							if (nInner>-1) {
								tree[j][k].locals+=M2LInner[l]*tree[j][nInner].multipoles;
								tree[j][k].locals+=boxLogHomogRadius[j]*tree[j][nInner].multipoles.sum()*Eigen::VectorXd::Ones(rank);
							}
						}
						//	Outer well-separated clusters
						for (int l=0; l<24; ++l) {
							int nOuter	=	tree[j][k].outerNumbers[l];
							if (nOuter>-1) {
								tree[j][k].locals+=M2LOuter[l]*tree[j][nOuter].multipoles;
								tree[j][k].locals+=boxLogHomogRadius[j]*tree[j][nOuter].multipoles.sum()*Eigen::VectorXd::Ones(rank);
							}
						}
					#endif
				}
			}
		}
	}




	void evaluate_list3() {
		for (int i=0; i<childless_boxes.size(); ++i) { // all childless boxes
			level_box lb = childless_boxes[i]; // box b
			for (int j=0; j<tree[lb.level][lb.box].list3.size(); ++j) { // all list3 boxes of a childless box
				level_box l3 = tree[lb.level][lb.box].list3[j];
				// list 3; level=l3.level; box = l3.box
				// evaluate cheb nodes of l3 box
				std::vector<pts2D> l3_ChebNodes;
				l3_ChebNodes	=	shift_scale_Cheb_Nodes(tree[l3.level][l3.box].center.x,tree[l3.level][l3.box].center.y,boxRadius[l3.level]);

				Eigen::MatrixXd Kernel_Matrix;
				obtain_List3_Operator(lb,l3_ChebNodes,Kernel_Matrix);
				tree[lb.level][lb.box].locals += Kernel_Matrix*tree[l3.level][l3.box].multipoles;
			}
		}
	}



//	Obtain the desired matrix for list 3
	void obtain_List3_Operator(level_box& lb, std::vector<pts2D>& l3_ChebNodes, Eigen::MatrixXd& T) { // lb is childless box
		T	=	Eigen::MatrixXd(rank,rank);
		for (int i=0; i<rank; ++i) {
			for (int j=0; j<rank; ++j) {
				T(i,j)	=	K->getInteraction(tree[lb.level][lb.box].chebNodes[i], l3_ChebNodes[j], a);
			}
		}
	}



	void evaluate_list4() {
		for (int l=2; l<=nLevels; ++l) { // all boxes // box b
			for (int b=0; b<nBoxesPerLevel[l]; ++b) {
				if (tree[l][b].exists) {
					for (int j=0; j<tree[l][b].list4.size(); ++j) { // all list3 boxes of a childless box
						level_box l4 = tree[l][b].list4[j];
						Eigen::MatrixXd Kernel_Matrix;
						// list 4; level=l4.level; box = l4.box
						// cheb nodes of l4 box already evaluated because l4 boxes are leaf nodes
						if (l!= nLevels) {
							std::vector<pts2D> boxb_ChebNodes;
							boxb_ChebNodes	=	shift_scale_Cheb_Nodes(tree[l][b].center.x,tree[l][b].center.y,boxRadius[l]);
							obtain_List4_Operator_1(boxb_ChebNodes,l4,Kernel_Matrix);
						}
						else { // childless box chebNodes already evaluated so avoid re-evaluating
							level_box lb;
							lb.level = l;
							lb.box = b;
							obtain_List4_Operator_2(lb,l4,Kernel_Matrix);
						}
						tree[l][b].locals += Kernel_Matrix*tree[l4.level][l4.box].multipoles;
					}
				}
			}
		}
	}




//	Obtain the desired matrix for list 4 of parent boxes
	void obtain_List4_Operator_1(std::vector<pts2D>& boxb_ChebNodes, level_box& lb, Eigen::MatrixXd& T) { // lb is childless box
		T	=	Eigen::MatrixXd(rank,rank);
		for (int i=0; i<rank; ++i) {
			for (int j=0; j<rank; ++j) {
				T(i,j)	=	K->getInteraction(boxb_ChebNodes[i], tree[lb.level][lb.box].chebNodes[j], a);
			}
		}
	}



//	Obtain the desired matrix for list 4 of childless boxes
	void obtain_List4_Operator_2(level_box& lb, level_box& l4, Eigen::MatrixXd& T) { // lb is childless box
		T	=	Eigen::MatrixXd(rank,rank);
		for (int i=0; i<rank; ++i) {
			for (int j=0; j<rank; ++j) {
				T(i,j)	=	K->getInteraction(tree[lb.level][lb.box].chebNodes[i], tree[l4.level][l4.box].chebNodes[j], a);
			}
		}
	}




	void evaluate_All_L2L() {
		for (int j=2; j<nLevels; ++j) {
			int J	=	j+1;
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				int K	=	4*k;
				if (tree[J][K].exists) { // if children exist
					tree[J][K].locals+=L2L[0]*tree[j][k].locals;
					tree[J][K+1].locals+=L2L[1]*tree[j][k].locals;
					tree[J][K+2].locals+=L2L[2]*tree[j][k].locals;
					tree[J][K+3].locals+=L2L[3]*tree[j][k].locals;
				}
			}
		}
	}


//	Self & Neighbor Interaction
	void evaluate_list1() {
		//tree[0][0].locals	=	Eigen::VectorXd::Zero(rank); // root
		for (int i=0; i<childless_boxes.size(); ++i) { // all childless boxes
			level_box lb = childless_boxes[i];
			for (int k=0; k<rank; ++k) { // box b chebNodes
				for (int j=0; j<tree[lb.level][lb.box].list1.size(); ++j) { // all list3 boxes of a childless box
					level_box l1 = tree[lb.level][lb.box].list1[j];
					for (int c=0; c<rank; ++c) { // box list1 chebNodes //equivalent to charge locations
						tree[lb.level][lb.box].locals(k) += tree[l1.level][l1.box].multipoles(c) * K->getInteraction(tree[l1.level][l1.box].chebNodes[c], tree[lb.level][lb.box].chebNodes[k], a);
					}
					




					/*for (int c=0; c<tree[l1.level][l1.box].charge_indices.size(); ++c) {
						pts2D charge_loc;
						charge_loc.x = charge_database[tree[l1.level][l1.box].charge_indices[c]].x;
						charge_loc.y = charge_database[tree[l1.level][l1.box].charge_indices[c]].y;
						tree[lb.level][lb.box].locals(k) += charge_database[tree[l1.level][l1.box].charge_indices[c]].q * K->getInteraction(charge_loc, tree[lb.level][lb.box].chebNodes[k], a);
					}*/
				}
			}
		}
	}





/*
	void evaluate_Leaf() {
		if (nLevels <2) {
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[nLevels]; ++k) {
				tree[nLevels][k].locals	=	Eigen::VectorXd::Zero(rank);
			}
		}
		#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[nLevels]; ++k) {
			for (int l=0; l<8; ++l) {
				int nNeighbor	=	tree[nLevels][k].neighborNumbers[l];
				if (nNeighbor > -1) {
					tree[nLevels][k].locals+=neighborInteraction[l]*tree[nLevels][nNeighbor].multipoles;
				}
			}
			tree[nLevels][k].locals+=selfInteraction*tree[nLevels][k].multipoles;
		}
	}

*/



	void perform_Error_Check() {
		int c 	=	rand()%childless_boxes.size();
		// for (int c=0; c<childless_boxes.size(); ++c) {

		level_box nBox = childless_boxes[c];
		
		Eigen::VectorXd potential	=	Eigen::VectorXd::Zero(rank);
		for (int l1=0; l1<rank; ++l1) {// cheb nodes of nBox
			for (int k=0; k<childless_boxes.size(); ++k) { // other boxes which includes nBox
				level_box other_boxes = childless_boxes[k];

				/*for (int l2=0; l2<tree[other_boxes.level][other_boxes.box].charge_indices.size(); ++l2) { // cheb nodes of other boxes or charge locations of charges in other boxes
					pts2D charge_loc;
					charge_loc.x = charge_database[tree[other_boxes.level][other_boxes.box].charge_indices[l2]].x;
					charge_loc.y = charge_database[tree[other_boxes.level][other_boxes.box].charge_indices[l2]].y;
					potential(l1)+=K->getInteraction(tree[nBox.level][nBox.box].chebNodes[l1], charge_loc, a)*charge_database[tree[other_boxes.level][other_boxes.box].charge_indices[l2]].q;
				}*/

				for (int l2=0; l2<rank; ++l2) { // cheb nodes of other boxes or charge locations of charges in other boxes (using multipoles)
					potential(l1)+=K->getInteraction(tree[nBox.level][nBox.box].chebNodes[l1], tree[other_boxes.level][other_boxes.box].chebNodes[l2], a)*tree[other_boxes.level][other_boxes.box].multipoles(l2);
				}
			}
		}
		Eigen::VectorXd error(rank);
		for (int k=0; k<rank; ++k) {
			error(k)	=	fabs((potential-tree[nBox.level][nBox.box].locals)(k)/potential(k));
		}
		cout << "nBox.level: " << nBox.level << " nBox.box: " << nBox.box << " er: "<< error.maxCoeff() << endl;
		// }
		
		//return error.maxCoeff();
	}


	void check4() {
		
		cout << "nn" << endl;
		for (int l=0; l<8; ++l) {
			cout << tree[2][0].neighborNumbers[l]<<",";
		}
		cout << endl << "in" << endl;
		for (int l=0; l<16; ++l) {
			cout << tree[2][0].innerNumbers[l]<<",";
		}
		cout << endl << "on" << endl;
		for (int l=0; l<16; ++l) {
			cout << tree[2][0].outerNumbers[l]<<",";
		}
		cout << endl;
	}

};

#endif
