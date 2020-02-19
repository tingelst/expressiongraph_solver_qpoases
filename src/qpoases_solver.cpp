#include <expressiongraph/qpoases_solver.hpp>
#include <expressiongraph/qpoases_messages.hpp>
#include <iostream>
namespace KDL {

qpOASESSolver::qpOASESSolver(int _nWSR, double _cputime, double _regularisation_factor)
  : regularisation_factor(_regularisation_factor), nWSR(_nWSR), cputime(_cputime) {
}

int qpOASESSolver::setupCommon(Context::Ptr _ctx) {
    ctx = _ctx;
    ndx_time = ctx->getScalarNdx("time");
    // construct index of all variables: nv_robot, nv_feature and the time variables (i.e. the state) :
    all_ndx.clear();
    ctx->getScalarsOfType("robot", all_ndx);
    nv_robot = all_ndx.size();
    ctx->getScalarsOfType("feature", all_ndx);
    nv_feature = all_ndx.size() - nv_robot;
    all_ndx.push_back(ndx_time);
    // allocate and initialize state:
    state.resize(all_ndx.size());
    for (size_t i = 0; i < all_ndx.size(); ++i) {
        VariableScalar* vs = ctx->getScalarStruct(all_ndx[i]);
        state[i] = vs->initial_value;
    }
    // add all expressions to the optimizer (context + output specification + monitors):
    expr_opt.prepare(all_ndx);
    ctx->addToOptimizer(expr_opt);
    // count and check the number of constraints for the different priority levels:
    nc_priority.resize(3);

    for (size_t i = 0; i < nc_priority.size(); ++i) {
        nc_priority[i] = 0;
    }
    for (size_t indx = 0; indx < ctx->cnstr_scalar.size(); ++indx) {
        ConstraintScalar& c = ctx->cnstr_scalar[indx];
        if (c.active) {
            if (c.priority >= (int)nc_priority.size()) {
                return -1;
            }
            nc_priority[c.priority] += 1;
        }
    }
    return 0;
}

void qpOASESSolver::reset() {
    H = Eigen::MatrixXd::Identity(nv, nv) * regularisation_factor;
    A = Eigen::MatrixXd::Zero(nc, nv);
    lb = Eigen::VectorXd::Constant(nv, -HUGE_VALUE);
    lbA = Eigen::VectorXd::Constant(nc, -HUGE_VALUE);
    ub = Eigen::VectorXd::Constant(nv, HUGE_VALUE);
    ubA = Eigen::VectorXd::Constant(nc, HUGE_VALUE);
    g = Eigen::VectorXd::Zero(nv);
}
void qpOASESSolver::setupMatrices(int _nv, int _nc) {
    nv = _nv;
    nc = _nc;
    reset();
    firsttime = true;
    if (nv != 0) {
        QP = qpOASES::SQProblem(nv, nc, qpOASES::HST_POSDEF);
    }
    QP.setPrintLevel(qpOASES::PL_NONE);
    solution.resize(nv);
}

void qpOASESSolver::printMatrices(std::ostream& os) {
    os << "H:\n" << H << "\n";
    os << "lb:\n" << lb.transpose() << "\n";
    os << "ub:\n" << ub.transpose() << "\n";
    os << "A:\n" << A << "\n";
    os << "lbA:\n" << lbA.transpose() << "\n";
    os << "ubA:\n" << ubA.transpose() << std::endl;
    os << "solution:\n" << solution.transpose() << std::endl;
    os << "state:\n" << state.transpose() << std::endl;
}

int qpOASESSolver::prepareExecution(Context::Ptr ctx) {
    int retval = setupCommon(ctx);
    if (retval != 0)
        return retval;
    initialization = false;
    // set up x variable as : robot | feature | slack
    optim_ndx.clear();
    ctx->getScalarsOfType("robot", optim_ndx);
    ctx->getScalarsOfType("feature", optim_ndx);
    setupMatrices(nv_robot + nv_feature + nc_priority[2], nc_priority[0] + nc_priority[1] + nc_priority[2]);
    return 0;
}

int qpOASESSolver::prepareInitialization(Context::Ptr ctx) {
    int retval = setupCommon(ctx);
    if (retval != 0)
        return retval;
    initialization = true;
    // set up x variable as : feature
    optim_ndx.clear();
    ctx->getScalarsOfType("feature", optim_ndx);
    setupMatrices(nv_feature + nc_priority[0], nc_priority[0]);
    return 0;
}

void qpOASESSolver::fillHessian() {
    // valid for both execution and initialization
    // optim_ndx contains the variable indices of the variables involved in the optimization (not including slack
    // variables)
    for (size_t i = 0; i < optim_ndx.size(); ++i) {
        VariableScalar* vs = ctx->getScalarStruct(optim_ndx[i]);
        H(i, i) = vs->weight->value() * regularisation_factor;
    }
}

void qpOASESSolver::fillConstraintScalar(const ConstraintScalar& c) {
    if (initialization && c.priority != 0)
        return;
    assert((0 <= cnr) && (cnr < nc));

    // called for updating the model
    double modelval = c.model->value();
    double modelder = c.model->derivative(ndx_time);
    double measval = c.meas->value();
    if (c.target_lower == -HUGE_VALUE) {
        lbA(cnr) = -HUGE_VALUE;
    } else {
        lbA(cnr) = c.controller_lower->compute(c.target_lower - measval, -modelder);
    }
    if (c.target_upper == HUGE_VALUE) {
        ubA(cnr) = HUGE_VALUE;
    } else {
        ubA(cnr) = c.controller_upper->compute(c.target_upper - measval, -modelder);
    }

    for (size_t i = 0; i < optim_ndx.size(); ++i) {
        A(cnr, i) = c.model->derivative(optim_ndx[i]);
    }

    if ((c.priority == 2) || (initialization && (c.priority == 0))) {
        // soft constraint: additional slack var:
        // assert( (0<=softcnr)&&(softcnr<nc_priority[2]));   not correct when initializing
        int j = optim_ndx.size() + softcnr;
        A(cnr, j) = 1.0;
        H(j, j) = regularisation_factor + c.weight->value();
        softcnr++;
    }
    cnr++;
}

void qpOASESSolver::fillConstraintBox(const ConstraintBox& c) {
    for (size_t i = 0; i < optim_ndx.size(); ++i) {
        if (optim_ndx[i] == c.variablenr) {
            lb[i] = c.target_lower;
            ub[i] = c.target_upper;
        }
    }
}

void qpOASESSolver::fill_constraints() {
    reset();
    softcnr = 0;
    cnr = 0;
    for (size_t indx = 0; indx < ctx->cnstr_scalar.size(); ++indx) {
        ConstraintScalar& c = ctx->cnstr_scalar[indx];
        if (c.active) {
            fillConstraintScalar(c);
        }
    }
    for (size_t indx = 0; indx < ctx->cnstr_box.size(); ++indx) {
        ConstraintBox& c = ctx->cnstr_box[indx];
        if (c.active) {
            fillConstraintBox(c);
        }
    }
}

void qpOASESSolver::setPrintLevel(int n) {
    QP.setPrintLevel((qpOASES::PrintLevel)n);
}

int qpOASESSolver::solve() {
    expr_opt.setInputValues(state);
    // without optimizer: ctx->setInputValues(all_ndx,state);
    if (nv == 0)
        return 0;
    fill_constraints();
    fillHessian();  // fill_constraints calls reset() that resets the values of H, so this should follow
                    // fill_constraints.
    // printMatrices(std::cout);
    int _nWSR = nWSR;
    double _cputime = cputime;
    int retval;
    if (firsttime) {
        retval = QP.init(H.data(), g.data(), A.data(), lb.data(), ub.data(), lbA.data(), ubA.data(), _nWSR, &_cputime);
        firsttime = false;
    } else {
        retval =
            QP.hotstart(H.data(), g.data(), A.data(), lb.data(), ub.data(), lbA.data(), ubA.data(), _nWSR, &_cputime);
    }
    if (retval != 0)
        return retval;
    retval = QP.getPrimalSolution(solution.data());
    // std::cout << solution.transpose() << std::endl;
    norm_change = 0;
    for (int i = 0; i < optim_ndx.size(); ++i) {
        // do not count the slack variables.
        norm_change += solution[i] * solution[i];
    }
    return retval;
}

double qpOASESSolver::getWeightedResult() {
    // std::cout << solution.transpose() << std::endl;
    double val = 0.0;
    for (int i = optim_ndx.size(); i < nv; ++i) {
        // std::cout << i << " ===> " << solution[i] << "\t\t" << H(i,i) << std::endl;
        val += solution[i] * solution[i] * H(i, i);
    }
    return val;
}

int qpOASESSolver::updateStep(double dt) {
    int retval = solve();
    if (retval != 0)
        return retval;
    if (initialization) {
        // solution contains feature
        // state    contains robot | feature | time  but (explicit) time remains constant.
        for (int i = 0; i < nv_feature; ++i) {
            state[nv_robot + i] += solution[i] * dt;
        }
    } else {
        // solution contains robot | feature | slack velocities
        // state    contains robot | feature | time
        for (int i = 0; i < nv_robot + nv_feature; ++i) {
            state[i] += solution[i] * dt;
        }
        state[nv_robot + nv_feature] += dt;
    }
    return 0;
}

std::string qpOASESSolver::errorMessage(int code) {
    int nrOfMessages = 0;
    while (qpoases_messages[nrOfMessages] != 0) {
        if (code == nrOfMessages) {
            return qpoases_messages[code];
        }
        nrOfMessages++;
    }
    std::stringstream ss;
    ss << "UNKNOWN ERRORCODE " << code;
    return ss.str();
}

std::string qpOASESSolver::getName() {
    return "qpOASES_velocity_resolution";
}

/*

// === NOT YET VERIFIED OR TESTED:
void qpOASESSolver::getJointNameToIndex(std::map<std::string,int>& namemap) {
    namemap.clear();
    for (int i=0;i<nv_robot;++i) {
        namemap[ ctx->getScalarStruct( all_ndx[i] )->name ] = i;
    }
}
void qpOASESSolver::getJointNameVector(std::vector<std::string>& namevec) {
    namevec.resize(nv_robot);
    for (int i=0;i<nv_robot;++i) {
        namevec[i] = ctx->getScalarStruct( all_ndx[i] )->name;
    }
}
void qpOASESSolver::getFeatureNameToIndex(std::map<std::string,int>& namemap) {
    namemap.clear();
    for (int i=0;i<nv_feature;++i) {
        namemap[ ctx->getScalarStruct( all_ndx[i+nv_robot] )->name ] = i;
    }
}
void qpOASESSolver::getFeatureNameVector(std::vector<std::string>& namevec) {
    namevec.resize(nv_feature);
    for (int i=0;i<nv_feature;++i) {
        namevec[i] = ctx->getScalarStruct( all_ndx[i+nv_robot] )->name;
    }
}



void qpOASESSolver::setFeatureStates(const Eigen::VectorXd& featstate) {
    assert( featstate.size() == nv_feature );
    for (int i=0;i<nv_feature;++i) {
        state[i+nv_robot] = featstate[i];
    }
}
void qpOASESSolver::getFeatureStates(Eigen::VectorXd& featstate) {
    assert( featstate.size() == nv_feature );
    for (int i=0;i<nv_feature;++i) {
        featstate[i] = state[i+nv_robot];
    }
}
void qpOASESSolver::setTime(double time) {
    state[nv_robot+nv_feature] = time;
}

double qpOASESSolver::getTime() {
    return state[nv_robot+nv_feature];
}

void qpOASESSolver::getJointVelocities(Eigen::VectorXd& _jntvel) {
    assert( !initialization );
    for (int i=0;i<nv_robot;++i) {
        _jntvel[i] = solution[i];
    }
}
void qpOASESSolver::getFeatureVelocities(Eigen::VectorXd& _featvel) {
    assert( !initialization );
    for (int i=0;i<nv_feature;++i) {
        _featvel[i] = solution[i+nv_robot];
    }
}
*/

/**
 * if you do not want to solve() but still want to evaluate all the
 * expressions
 */
void qpOASESSolver::evaluate_expressions() {
    expr_opt.setInputValues(state);
    solution.setZero(nv);  // fill zero in the solution vector
}

void qpOASESSolver::getJointNameToIndex(std::map<std::string, int>& namemap) {
    namemap.clear();
    for (int i = 0; i < nv_robot; ++i) {
        namemap[ctx->getScalarStruct(all_ndx[i])->name] = i;
    }
}

void qpOASESSolver::getJointNameVector(std::vector<std::string>& namevec) {
    namevec.resize(nv_robot);
    for (int i = 0; i < nv_robot; ++i) {
        namevec[i] = ctx->getScalarStruct(all_ndx[i])->name;
    }
}

void qpOASESSolver::getFeatureNameToIndex(std::map<std::string, int>& namemap) {
    namemap.clear();
    for (int i = 0; i < nv_feature; ++i) {
        namemap[ctx->getScalarStruct(all_ndx[i + nv_robot])->name] = i;
    }
}

void qpOASESSolver::getFeatureNameVector(std::vector<std::string>& namevec) {
    namevec.resize(nv_feature);
    for (int i = 0; i < nv_feature; ++i) {
        namevec[i] = ctx->getScalarStruct(all_ndx[i + nv_robot])->name;
    }
}

int qpOASESSolver::getNrOfJointStates() {
    return nv_robot;
}

int qpOASESSolver::getNrOfFeatureStates() {
    return nv_feature;
}

void qpOASESSolver::setJointStates(const Eigen::VectorXd& jntstate) {
    assert(jntstate.size() == nv_robot);
    for (int i = 0; i < nv_robot; ++i) {
        state[i] = jntstate[i];
    }
}

void qpOASESSolver::getJointStates(Eigen::VectorXd& jntstate) {
    assert(jntstate.size() == nv_robot);
    for (int i = 0; i < nv_robot; ++i) {
        jntstate[i] = state[i];
    }
}

void qpOASESSolver::setAndUpdateJointStates(const Eigen::VectorXd& jntstate) {
    assert(jntstate.size() == nv_robot);
    for (int i = 0; i < nv_robot; ++i) {
        state[i] = jntstate[i];
    }
    expr_opt.setInputValues(state);
}

void qpOASESSolver::setFeatureStates(const Eigen::VectorXd& featstate) {
    assert(featstate.size() == nv_feature);
    for (int i = 0; i < nv_feature; ++i) {
        state[i + nv_robot] = featstate[i];
    }
}

void qpOASESSolver::getFeatureStates(Eigen::VectorXd& featstate) {
    assert(featstate.size() == nv_feature);
    for (int i = 0; i < nv_feature; ++i) {
        featstate[i] = state[i + nv_robot];
    }
}

void qpOASESSolver::setTime(double time) {
    state[nv_robot + nv_feature] = time;
}

double qpOASESSolver::getTime() {
    return state[nv_robot + nv_feature];
}

void qpOASESSolver::getJointVelocities(Eigen::VectorXd& _jntvel) {
    assert(!initialization);
    for (int i = 0; i < nv_robot; ++i) {
        _jntvel[i] = solution[i];
    }
}

void qpOASESSolver::getFeatureVelocities(Eigen::VectorXd& _featvel) {
    assert(!initialization);
    for (int i = 0; i < nv_feature; ++i) {
        _featvel[i] = solution[i + nv_robot];
    }
}

double qpOASESSolver::getNormChange() {
    // assert( initialization );
    return sqrt(norm_change);
}

void qpOASESSolver::setState(const Eigen::VectorXd& _state) {
    state = _state;
}

void qpOASESSolver::getState(Eigen::VectorXd& _state) {
    _state = state;
}

void qpOASESSolver::setInitialValues() {
    for (size_t i = 0; i < all_ndx.size(); ++i) {
        VariableScalar* vs = ctx->getScalarStruct(all_ndx[i]);
        vs->initial_value = state[i];
    }
}

}  // end of namespace KDL
