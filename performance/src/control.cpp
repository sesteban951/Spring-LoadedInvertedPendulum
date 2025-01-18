#include "../inc/control.h"

Controller::Controller(YAML::Node config_file)
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
    // initialize the input trajectory bundle
    Vector_2d_Traj_Bundle U_bundle;
    U_bundle.resize(K);

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
