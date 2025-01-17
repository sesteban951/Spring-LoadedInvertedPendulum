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
Vector_d_Traj Controller::sample_input_trajectory(int K)
{
    // initialize the input trajectory bundle
    Vector_d_Traj_Bundle U_bundle;
    U_bundle.resize(K);

    std::cout << "1" << std::endl;

    // sample the input trajectories
    Vector_d mean = this->dist.mean;
    Matrix_d Sigma = this->dist.cov;

    std::cout << "mean size: " << mean.size() << std::endl;
    std::cout << "Sigma size: " << Sigma.rows() << "x" << Sigma.cols() << std::endl;

    // perform cholesky decomposition 
    Eigen::LLT<Matrix_d> llt(Sigma);  
    Matrix_d L = llt.matrixL();  

    std::cout << "2" << std::endl;
    std::cout << "L size: " << L.rows() << "x" << L.cols() << std::endl;

    // Generate a standard normal vector
    Vector_d Z = Vector_d::Zero(mean.size());
    Z.resize(mean.size());

    for (int i = 0; i < mean.size(); i++) {
        Z(i) = this->normal_dist(this->rand_generator);
    }

    std::cout << "Z: " << Z.size() << std::endl;
    std::cout << "3" << std::endl;

    // Generate a sample from the distribution
    Vector_d Ut;
    Ut.resize(mean.size());

    std::cout << "Ut: " << Ut.size() << std::endl;

    Ut = mean + L * Z;
    Vector_d_Traj Ut_Traj;
    Ut_Traj.resize(this->params.Nu);
    for (int i = 0; i < this->params.Nu; i++) {
        Ut_Traj[i] = Ut;
    }

    std::cout << "4" << std::endl;

    for (int i = 0; i < K; i++) {
        std::cout << "Ut_Traj: " << Ut_Traj[i].transpose() << std::endl;
    }

    return Ut_Traj;
}
