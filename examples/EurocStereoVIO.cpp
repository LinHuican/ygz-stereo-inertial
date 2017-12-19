/**
 * This is the Euroc stereo visual-inertial odometry program
 * Please specify the dataset directory in the config file
*/

#include <opencv2/opencv.hpp>

#include "ygz/System.h"
#include "ygz/EurocReader.h"

#include <cv_bridge/cv_bridge.h> // added by huicanlin

using namespace std;
using namespace ygz;

void SaveTrajectoryTUM(vector<SE3d> campose, vector<double> vTimeStamp);//added by huicanlin
Eigen::Matrix<double,3,3> ConvertertoMatrix3d(const cv::Mat &cvMat3); // added by huicanlin
std::vector<float> ConvertertoQuaternion(const cv::Mat &M); // added by huicanlin

int main(int argc, char **argv) {

    if (argc != 2) {
        LOG(INFO) << "Usage: EurocStereoVIO path_to_config" << endl;
        return 1;
    }

    FLAGS_logtostderr = true;//干吗的？
    google::InitGoogleLogging(argv[0]);

    string configFile(argv[1]);
    cv::FileStorage fsSettings(configFile, cv::FileStorage::READ);

    if (fsSettings.isOpened() == false) {
        LOG(FATAL) << "Cannot load the config file from " << argv[1] << endl;
    }

    System system(argv[1]);

    // rectification parameters
    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() ||
        D_r.empty() ||
        rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return 1;
    }

    cv::Mat M1l, M2l, M1r, M2r;
    cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, M1l,
                                M2l);
    cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F, M1r,
                                M2r);

    string leftFolder = fsSettings["LeftFolder"];
    string rightFolder = fsSettings["RightFolder"];
    string imuFolder = fsSettings["IMUFolder"];
    string timeFolder = fsSettings["TimeFolder"];

    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimeStamp;
    if (LoadImages(leftFolder, rightFolder, timeFolder, vstrImageLeft, vstrImageRight, vTimeStamp) == false)
        return 1;

    VecIMU vimus;
    if (LoadImus(imuFolder, vimus) == false)
        return 1;

    // read TBC
    cv::Mat Rbc, tbc;
    fsSettings["RBC"] >> Rbc;
    fsSettings["TBC"] >> tbc;
    if (!Rbc.empty() && tbc.empty()) {
        Matrix3d Rbc_;
        Vector3d tbc_;
        Rbc_ <<
             Rbc.at<double>(0, 0), Rbc.at<double>(0, 1), Rbc.at<double>(0, 2),
                Rbc.at<double>(1, 0), Rbc.at<double>(1, 1), Rbc.at<double>(1, 2),
                Rbc.at<double>(2, 0), Rbc.at<double>(2, 1), Rbc.at<double>(2, 2);
        tbc_ <<
             tbc.at<double>(0, 0), tbc.at<double>(1, 0), tbc.at<double>(2, 0);

        setting::TBC = SE3d(Rbc_, tbc_);
    }

    vector<Eigen::Matrix4d> campose;
    vector<double> camtime;
    size_t imuIndex = 0;
    for (size_t i = 0; i < vstrImageLeft.size(); i++) {
        cv::Mat imLeft, imRight, imLeftRect, imRightRect;

        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[i], CV_LOAD_IMAGE_UNCHANGED);// vstrImageLeft： 图片的文件名
        imRight = cv::imread(vstrImageRight[i], CV_LOAD_IMAGE_UNCHANGED);

        if (imLeft.empty() || imRight.empty()) {
            LOG(WARNING) << "Cannot load image " << i << endl;
            continue;
        }

        cv::remap(imLeft, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(imRight, imRightRect, M1r, M2r, cv::INTER_LINEAR);

        // and imu
        VecIMU vimu;

        double tframe = vTimeStamp[i];

        while (1) {
            const ygz::IMUData &imudata = vimus[imuIndex];
            if (imudata.mfTimeStamp >= tframe)
                break;
            vimu.push_back(imudata);
            imuIndex++;
        }

        Matrix4d Twb = system.AddStereoIMU(imLeftRect, imRightRect, tframe, vimu).matrix();
        campose.push_back(Twb);
        camtime.push_back(vTimeStamp[i]);//跟踪失败的情形怎么办？
    }

    cout << endl << "saving trajectory ... " << endl;
    ofstream f;
    f.open("VIOtrajectoryTUM");
    f << fixed;
    Matrix3d Rwb;
    Vector3d twb;
    //for(uint k = 0; k < vTimeStamp.size(); k++)
    for(uint k = 0; k < camtime.size(); k++)
    {
        twb = Vector3d(campose[k](0,3), campose[k](1,3), campose[k](2,3));
        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                Rwb(i,j) = campose[k](i,j);
        Eigen::Quaterniond q(Rwb);
        f << setprecision(0) << (camtime[k])*1e9 << " " <<  setprecision(6) << twb[0] << " " << twb[1] << " " << twb[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;//修改
    }

    f.close();
    cout << endl << "trajectory saved!" << endl;

    return 0;
}

void SaveTrajectoryTUM(vector<SE3d> campose, vector<double> vTimeStamp)
{
    /*
    //SaveTrajectoryTUM();
    ofstream f;
    //f.open(filename.c_str());
    f.open("SaveTrajectoryTUM");
    f << fixed;
    for(uint i=0; i<vTimeStamp.size(); i++)
    {
        campose[i][0];
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        //Rwc = campose[i].rowRange(0,3).colRange(0,3).t();
        //twc = -Rwc * campose[i].rowRange(0,3).col(3);
        vector<float> q = ConvertertoQuaternion(Rwc);
        //std::setp
        //f << setprecision(0) << (vTimeStamp[i])*1e9 << " " <<  setprecision(6) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;//修改
    }
    */
}

Eigen::Matrix<double,3,3> ConvertertoMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

std::vector<float> ConvertertoQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double,3,3> eigMat = ConvertertoMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}
