#include "../inc/control.h"

Controller::Controller(YAML::Node config_file) : dynamics(config_file)
{
    // set the control parameters
    this->params.N = config_file["CTRL_PARAMS"]["N"].as<int>();
    this->params.dt = config_file["CTRL_PARAMS"]["dt"].as<double>();
    this->params.K = config_file["CTRL_PARAMS"]["K"].as<int>();
    this->params.Nu = config_file["CTRL_PARAMS"]["Nu"].as<int>();
    this->params.N_elite = config_file["CTRL_PARAMS"]["N_elite"].as<int>();
    this->params.CEM_iters = config_file["CTRL_PARAMS"]["CEM_iters"].as<int>();
    
    // build the cost matrices from the diagonal elements
    std::vector<double> Q_diags_temp = config_file["CTRL_PARAMS"]["Q_diags"].as<std::vector<double>>();
    std::vector<double> Qf_diags_temp = config_file["CTRL_PARAMS"]["Qf_diags"].as<std::vector<double>>();
    std::vector<double> R_diags_temp = config_file["CTRL_PARAMS"]["R_diags"].as<std::vector<double>>();

    this->params.Q  = Matrix_8d::Zero();
    this->params.Qf = Matrix_8d::Zero();
    this->params.R = Matrix_2d::Zero();

    for (int i = 0; i < Q_diags_temp.size(); i++) {
        this->params.Q(i, i) = Q_diags_temp[i];
        this->params.Qf(i, i) = Qf_diags_temp[i];
    }
    
    for (int i = 0; i < R_diags_temp.size(); i++) {
        this->params.R(i, i) = R_diags_temp[i];
    }

    // construct the initial distribution
    this->initialize_distribution(config_file);
}


// construct the intial distribution
void Controller::initialize_distribution(YAML::Node config_file)
{
    // initialize the matrices
    this->dist.mean.resize(this->params.Nu * 2);
    this->dist.cov.resize(this->params.Nu * 2, this->params.Nu * 2);
    this->dist.mean.setZero();
    this->dist.cov.setZero();

    // set the initial mean
    std::vector<double> mean_temp = config_file["DIST_PARAMS"]["mu"].as<std::vector<double>>();
    Vector_2d mean;
    mean << mean_temp[0], mean_temp[1];
    for (int i = 0; i < this->params.Nu; i++) {
        this->dist.mean.segment<2>(2 * i) = mean;
    }

    // set the initial covariance
    std::vector<double> cov_temp = config_file["DIST_PARAMS"]["sigma"].as<std::vector<double>>();
    Matrix_2d cov;
    cov << cov_temp[0], 0.0,
           0.0, cov_temp[1];
    for (int i = 0; i < this->params.Nu; i++) {
        this->dist.cov.block<2, 2>(2 * i, 2 * i) = cov;
    }

    // set if covariance should be strictly diagonal
    this->dist.diag_cov = config_file["DIST_PARAMS"]["diag_cov"].as<bool>();

    // set the random 
    this->dist.seed = config_file["DIST_PARAMS"]["seed"].as<int>();
    this->dist.seed_enabled = config_file["DIST_PARAMS"]["seed_enabled"].as<bool>();

    // create random device
    std::random_device rand_device;

    // use the random device to seed Mersenne Twister generator
    std::mt19937 rand_generator(rand_device());

    // set the seed if enabled
    if (this->dist.seed_enabled) {
        rand_generator.seed(this->dist.seed);
    }

    // Create a normal distribution
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    
    // set the random number generator and normal distribution
    this->rand_generator = rand_generator;
    this->normal_dist = normal_dist;
}


// sample input trajectories
Vector_2d_Traj_Bundle Controller::sample_input_trajectory(int K)
{
    // sample the input trajectories
    Vector_d mu = this->dist.mean;
    Matrix_d Sigma = this->dist.cov;

    // perform cholesky decomposition 
    Eigen::LLT<Matrix_d> llt(Sigma);  
    Matrix_d L = llt.matrixL();  

    // check if the covariance is positive definite
    if (llt.info() == Eigen::NumericalIssue) {
        throw std::runtime_error("Covariance matrix is possibly not positive definite");
    }

    // Generate random input trajectories and store them in the input bundle
    Vector_d Z_vec, U_vec;
    Vector_2d_Traj U_traj;
    Vector_2d U_t;
    Z_vec.resize(mu.size());
    U_vec.resize(mu.size());
    U_traj.resize(this->params.Nu);

    // U ~ N(mu, Sigma) <=> U = L * Z + mu; Z ~ N(0, I)
    // initialize the input trajectory bundle
    Vector_2d_Traj_Bundle U_bundle;
    U_bundle.resize(K);
    for (int i = 0; i < K; i++) {
        // populate the Z vector
        for (int i = 0; i < mu.size(); i++) {
            Z_vec(i) = this->normal_dist(this->rand_generator);
        }

        // generate the input trajectory
        U_vec = L * Z_vec + mu;

        // flatten U_vec into U_traj
        for (int j = 0; j < this->params.Nu; j++) {
            U_t = U_vec.segment<2>(2 * j);
            U_traj[j] = U_t;
        }

        // store the input trajectory
        U_bundle[i] = U_traj;
    }

    return U_bundle;
}


// compute mean and covariance from a bundle of control inputs
void Controller::update_dsitribution_params(Vector_2d_Traj_Bundle U_bundle)
{
    // initialize the mean and covariance
    Vector_d mean;
    Matrix_d cov;
    mean.resize(this->params.Nu * 2);
    cov.resize(this->params.Nu * 2, this->params.Nu * 2);

    // size of the bundle
    int K = U_bundle.size(); // not necceseraly equal to this->params.K

    // used for computing the mean
    Matrix_d U_data;
    U_data.resize(this->params.Nu * 2, K);

    // compute the mean
    Vector_2d_Traj U_traj;
    Vector_d U_traj_vec;
    U_traj_vec.resize(this->params.Nu * 2);
    for (int i = 0; i < K; i++) {

        // vectorize the input trajectory
        U_traj = U_bundle[i];
        for (int j = 0; j < this->params.Nu; j++) {
            U_traj_vec.segment<2>(2 * j) = U_traj[j];
        }
        mean += U_traj_vec;

        // insert into data matrix to use later
        U_data.col(i) = U_traj_vec;
    }
    mean /= K;

    // compute the sample covariance (K-1 b/c Bessel correction)
    cov = (1.0 / (K-1)) * (U_data.colwise() - mean) * (U_data.colwise() - mean).transpose();

    // TODO: Add an epsilon to eigenvalues for numerical stability
    // TODO: Add stricly diagonal covariance option
    // TODO: Add lower bounding of the covariance

    // update the distribution
    this->dist.mean = mean;
    this->dist.cov = cov;    
}


// generate a reference trajectory for the predictive control to track
Vector_8d_Traj Controller::generate_reference_trajectory(Vector_4d x0_com)
{
    // pass reference to the dynamics
    Vector_8d_Traj X_ref;
    X_ref.resize(this->params.N);

    // populate the reference trajectory
    Vector_8d xi_ref;
    double vx = 0.0;
    for (int i = 0; i < this->params.N; i++) {
        xi_ref << x0_com(0) + vx * this->params.dt,  // px
                  0.7,                               // pz
                  vx,                                // vx
                  0.0,                               // vz
                  0.65,                              // l0
                  0.0,                               // theta
                  0.0,                               // rdot
                  0.0;                               // thetadot

        // insert into trajectory
        X_ref[i] = xi_ref;
    }

    return X_ref;
}


// evaulate the cost function given a solution
double Controller::cost_function(Vector_8d_Traj X_ref, Solution Sol, Vector_2d_Traj U)
{
    // trajectory length 
    int N = this->params.N;

    // upack the relevant variables
    Vector_6d_Traj X_sys = Sol.x_sys_t;
    Vector_4d_Traj X_leg = Sol.x_leg_t;

    // convert to matrix
    Matrix_d X_sys_mat, X_leg_mat;
    X_sys_mat.resize(6, N);
    X_leg_mat.resize(4, N);
    for (int i = 0; i < N; i++) {
        X_sys_mat.col(i) = X_sys[i];
        X_leg_mat.col(i) = X_leg[i];
    }

    // combine the COM state with the leg state
    Matrix_d X_com;
    X_com.resize(4, N);
    X_com = X_sys_mat.block(0, 0, 4, N);
    
    Matrix_d X;
    X.resize(8, N);
    X << X_com, X_leg_mat;

    // state cost
    Vector_8d xi, xi_des, ei;
    double J_state, cost;
    J_state = 0.0;
    for (int i = 0; i < N; i++) {
        // compute error state
        xi = X.col(i);
        xi_des = X_ref[i];
        ei = (xi - xi_des);

        // compute the stage cost
        cost = ei.transpose() * this->params.Q * ei;
        J_state += cost;
    }

    // terminal cost
    xi = X.col(N - 1);
    xi_des = X_ref[N - 1];
    ei = (xi - xi_des);
    cost = ei.transpose() * this->params.Qf * ei;
    J_state += cost;

    // total cost
    double J_total; 
    J_total = J_state;

    return J_total;
}


// perform open loop rollouts
MC_Tuple Controller::monte_carlo(Vector_6d x0_sys, Vector_2d p0_foot, Domain d0)
{
    // compute u(t) dt (N of the integration is not necessarily equal to the number of control points)
    double T = (this->params.N-1) * this->params.dt;
    double dt_u = T / (this->params.Nu-1);

    auto t0 = std::chrono::high_resolution_clock::now();
    // generate the time arrays
    Vector_1d_Traj T_x;
    Vector_1d_Traj T_u;
    T_x.resize(this->params.N);
    T_u.resize(this->params.Nu);
    for (int i = 0; i < this->params.N; i++) {
        T_x[i] = i * this->params.dt;
    }
    for (int i = 0; i < this->params.Nu; i++) {
        T_u[i] = i * dt_u;
    }
    auto tf = std::chrono::high_resolution_clock::now();
    std::cout << "Time to generate time arrays: " << std::chrono::duration_cast<std::chrono::microseconds>(tf - t0).count() << " microseconds" << std::endl;

    // generate bundle of input trajectories
    Vector_2d_Traj_Bundle U_bundle;
    t0 = std::chrono::high_resolution_clock::now();
    U_bundle = this->sample_input_trajectory(this->params.K);
    tf = std::chrono::high_resolution_clock::now();
    std::cout << "Time to sample input trajectories: " << std::chrono::duration_cast<std::chrono::microseconds>(tf - t0).count() << " microseconds" << std::endl;

    // initialize the containers for the solutions
    Solution_Bundle Sol_bundle;
    Vector_1d_Traj J;
    J.resize(U_bundle.size());
    Sol_bundle.resize(U_bundle.size());

    // generate the reference trajectory
    Vector_8d_Traj X_ref;
    t0 = std::chrono::high_resolution_clock::now();
    X_ref = this->generate_reference_trajectory(x0_sys.head<4>());
    tf = std::chrono::high_resolution_clock::now();
    std::cout << "Time to generate reference trajectory: " << std::chrono::duration_cast<std::chrono::microseconds>(tf - t0).count() << " microseconds" << std::endl;

    // loop over the input trajectories
    Solution sol;
    t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < U_bundle.size(); k++) {

        // perform the rollout
        sol = this->dynamics.RK3_rollout(T_x, T_u, x0_sys, p0_foot, d0, U_bundle[k]);

        // compute the cost
        J[k] = this->cost_function(X_ref, sol, U_bundle[k]);

        // store the solution
        Sol_bundle[k] = sol;
    }
    tf = std::chrono::high_resolution_clock::now();
    std::cout << "Time to perform rollouts: " << std::chrono::duration_cast<std::chrono::microseconds>(tf - t0).count() << " microseconds" << std::endl;

    // pack solutions into a tuple
    MC_Tuple mc_tuple;
    mc_tuple = std::make_tuple(Sol_bundle, U_bundle, J);

    // return the solutions
    return mc_tuple;
}
