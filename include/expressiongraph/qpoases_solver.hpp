#ifndef EXPRESSIONGRAPH_TF_QPOASES_SOLVER_HPP
#define EXPRESSIONGRAPH_TF_QPOASES_SOLVER_HPP

#include <expressiongraph/context.hpp>
#include <qpOASES/SQProblem.hpp>
#include <Eigen/Dense>
#include <kdl/conversions.hpp>
#include <expressiongraph/solver.hpp>

namespace KDL {

#define HUGE_VALUE 1e20

/**
 * This solver understands three priorities:
 *   - 0:  hard constraint, used in initial loop closure.
 *   - 1:  hard constraint.
 *   - 2:  soft constraint.
 *
 * The solver expects three types of variables:
 *   - robot : the actuated variables
 *   - feature : feature variables
 *   - time    : type of the time variable.
 * There should be one variable defined with name "time".
 *
 * There are two different sets of variables:
 *  - There is a state, representing everything that to represent the state (including explicit time).
 *    This is used to  set the input values for all expression graphs.
 *  - There is the set of optimized variables. This is used by the QP-solver.
 * \nosubgrouping
 */
class qpOASESSolver :public solver{
protected:
        // reset the problem before filling it in:
        void reset();
        /// auxiliary methods:
        /// @{
        /// common set-up 
        /// \return 0 if successful
        int setupCommon(Context::Ptr _ctx);
        /// set-up qpOASES solver and matrices with the appropriate sizes
        /// \param [in] nv number of variables
        /// \param [in] nc number of constraints
        void setupMatrices(int nv, int nc);

        /// fill in the appropriate values into the Hessian
        void fillHessian();
        /// fill in a ConstraintScalar into the matrices of the optimization problem 
        void fillConstraintScalar(const ConstraintScalar& c);
        /// fill in a ConstraintBox into the matrices of the optimization problem 
        void fillConstraintBox(const ConstraintBox& c);
        /// fill in all constraints into the matrices of the optimization problem
        void fill_constraints();
        /// @}
        
        /// member variables (protected access)
        /// @{
        

		std::vector<int>    nc_priority;        ///< number of constraints for each priority number:
		int                 nc;                 ///< total number of constraints
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> A;    ///< constraint matrix  A
        Eigen::VectorXd     lb;                 ///< lb <= x
        Eigen::VectorXd     ub;                 ///< x <= ub
        Eigen::VectorXd     lbA;                ///< lba <= A*x
        Eigen::VectorXd     ubA;                ///< A*x <= ubA
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> H;    ///< optimize x'*H*x + g'*x
        Eigen::VectorXd     g;                  ///< optimize x'*H*x + g'*x
		qpOASES::SQProblem  QP;                 ///< QP Solver
        
		int                 softcnr;            ///<  counter to fill in the soft constraints.
        int                 cnr;                ///<  counter to fill in (all) constraints.

		double              regularisation_factor;   ///< regularisation factor double regularisation_factor;
        int                 nWSR;               ///< maximum number of solver iterations. 
        double              cputime;            ///< maximum execution time.

        /// @}
public:
    typedef boost::shared_ptr< qpOASESSolver > Ptr;

    /**
     * \param [in]  nWSR  maximum number of solver iterations.
     * \param [in]  cputime maximum execution time. If this value is equal to zero, it is ignored.
     * \param [in]  regularisation_factor   regularisation_factor to be used.
     */
    qpOASESSolver(int nWSR,double cputime, double regularisation_factor);

    /// set the qpOASES print level
    void setPrintLevel(int n);

    /**
     * Prepare for solving during an iterative initialization.
     * \param [in] ctx Context to initialize  the solver for.
     * \param [in] time time to start execution with.
     * \return 0 if sucessful, returns:
     *    - -1 if an unknown priority level is used.
     */
    virtual int prepareInitialization(Context::Ptr ctx);

    /**
     * set the initial values in the context to the current state.
     * (typically done after an initialization run).
     */    


    void printMatrices(std::ostream& os);

    /**
     * Prepare for solving during an iterative execution.
     * \param [in] ctx Context to initialize  the solver for.
     * \param [in] time time to start execution with.
     * \return 0 if sucessful, returns:
     *    - -1 if an unknown priority level is used.
     */
    virtual int prepareExecution(Context::Ptr ctx);



    /**
     * solves the optimization problem.
     * The typical call pattern is: 
     \code{.cpp} 
     \endcode 
     * \return 0 if successfull, use error_message method to interprete the error codes.
     */
    virtual int solve();

    virtual double getWeightedResult();





    virtual int  updateStep(double dt);
    
    /**
     * returns a description of the error codes returned by solve().
     */
    virtual  std::string errorMessage(int code) ;

    virtual std::string getName();

	virtual void evaluate_expressions();
   
	virtual void getJointNameToIndex(std::map<std::string,int>& namemap);
	virtual void getJointNameVector(std::vector<std::string>& namevec);
	virtual void getFeatureNameToIndex(std::map<std::string,int>& namemap);
	virtual void getFeatureNameVector(std::vector<std::string>& namevec);
	virtual int  getNrOfJointStates();
	virtual int  getNrOfFeatureStates();
 
    /**
     * /caveat jntstate should have the correct size ( nr of joints).
     */
	virtual void setJointStates(const Eigen::VectorXd& jntstate);
	virtual void getJointStates(Eigen::VectorXd& jntstate);

    /**
     * /caveat jntstate should have the correct size ( nr of joints).
     * works as setJointStates, but also updates the relations on which constraints are expressed.
     * after this call is possible to ask for jacobians from expressions
     */
	virtual void setAndUpdateJointStates(const Eigen::VectorXd& jntstate);

    /**
     * /caveat featstate should have the correct size ( nr of feature variables).
     */
	virtual void setFeatureStates(const Eigen::VectorXd& featstate);
	virtual void getFeatureStates(Eigen::VectorXd& featstate);
	virtual void setTime( double time);
	virtual double getTime();

    /**
     * /caveat not to be called during the initialization phase.
     */
	virtual void getJointVelocities(Eigen::VectorXd& _jntvel);

    /**
     * /caveat not to be called during the initialization phase.
     */
	virtual void getFeatureVelocities(Eigen::VectorXd& _featvel);


    /**
     * returns the norm af the relevant state, ie feature variables for an initialisation step,
     * robot + feature variables for execution step.
     * only valid to call during initialization.
     */
	virtual double getNormChange();
	
    /**
     * Sets the state variable (robot | feature | time) to the given values.
     */
	virtual void setState(const Eigen::VectorXd& _state);

    /**
     * Gets the state variable (robot | feature | time).
     */
	virtual void getState(Eigen::VectorXd& _state); 


    /**
     * set the initial values in the context to the current state.
     * (typically done after an initialization run).
     */    
	virtual void setInitialValues();

    virtual ~qpOASESSolver() {}
};

} // namespace KDL
#endif

