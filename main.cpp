/**
Pedro Martins, FCT-UNL, 2018

Aplication features:
1 - Train SVM classifiers.
2 - Exponential grid-search.
4 - Use own training/evaluation ratio for cross validation or,
5 - Use opencv SVM trainAuto with automatic optimal parameters
3 - Automatic evaluation of classifiers performance.
**/

#include <opencv2/ml.hpp>
#include <fstream>
#include <chrono>

#define argb(name) cmd.get<bool>(name)
#define arg(name) cmd.get<string>(name)
#define argi(name) cmd.get<int>(name)

using namespace std;
using namespace cv;
using namespace cv::ml;

const char *binary_scores = "";
const char *feature_vector = "";

double c, g;
int total_items, numberOfFeatures, numberToTrain;
int evaluation_items = 10;
bool automatic = false;

vector< vector<double> > featureTrainingData, featureEvaluationData, featureAutoTrainingData;
vector< int > binaryTrainingData, binaryEvaluationData, binaryAutoTrainingData;

//Exponential search parameters
const float cExpFactor = 0.01;
const float gExpFactor = 0.0000000001;
const float cExpMax = 500.0;
const float gExpMax = 1.0;
const double multiplier = 2;

///methods
string printHelp();
void countData();
void loadData(int ev_start, int ev_end);
void loadBinData(int ev_start, int ev_end );


int main(int argc, const char **argv)
{
    auto start = chrono::high_resolution_clock::now();
    cout << "OpenCV SVM tool.\n";
    try
    {
        const char *keys =
            "{ e | | }"
            "{ a | | }"
            "{ h | | }";
        CommandLineParser cmd(argc, argv, keys);

        ///parse the arguments

        if(cmd.has("h") || argc < 2)
        {
            cout << printHelp()<<endl;
            return 0;
        }
        if(cmd.has("a")) automatic = argb("a");
        if(cmd.has("e")) evaluation_items = argi("e");

        feature_vector = argv[1];
        binary_scores = argv[2];

        countData();

        int evaluation_start = 0;
        int evaluation_end = evaluation_items;
        int numberToTrain = total_items - evaluation_items;

        cout << total_items <<" samples, "<< numberOfFeatures <<" features, "
             << total_items/evaluation_items <<" folds, "<<" auto: "<< automatic<<endl;

        ///Initialize vectors
        featureTrainingData.assign(numberToTrain, vector<double>(numberOfFeatures,0));
        binaryTrainingData.assign(numberToTrain,0);
        featureEvaluationData.assign(evaluation_items, vector<double>(numberOfFeatures,0));
        binaryEvaluationData.assign(evaluation_items,0);

        /// Create the SVM ///////////////////////////////////////////////////
        Ptr<SVM> svm = SVM::create();
        svm->setType(SVM::C_SVC);
        //LINEAR POLY RBF SIGMOID CHI2 INTER
        svm->setKernel(SVM::RBF);
        //svm->setNu(0.6);
        //svm->setP(0);
        //svm->setCoef0(0);
        //svm->setDegree(2);

        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 0.000001));
        Mat C = (Mat_<double>(1,2) << 1, 1);
        svm->setClassWeights(C);
        cout<<"class weights: "<<svm->getClassWeights()<<endl;

        if(!automatic)
        {
            double opt_acertos = 0.0;
            double opt_c = 0.0;
            double opt_g = 0.0;

            c = cExpFactor;
            g = gExpFactor;


            ///cycle and search optimal C,Gamma with cross validation
            while(c <= cExpMax)
            {
                while(g <= gExpMax)
                {
                    double crossAccuracy = 0.0;
                    double crossPrecision = 0.0;
                    double crossRecall = 0.0;
                    double crossFPR = 0.0;
                    double crossFNR = 0.0;
                    int folds = 0;

                    while(evaluation_start < total_items) //cycle n folds
                    {

                        loadBinData(evaluation_start, evaluation_end );
                        loadData(evaluation_start, evaluation_end );

                        ///Setup the input matrixes for training and evaluate the svm /////////////
                        Mat trainingDataMat = Mat(numberToTrain, numberOfFeatures, CV_32FC1);

                        for(int i=0; i<numberToTrain; ++i)
                            for(int j=0; j<numberOfFeatures; ++j)
                            {
                                trainingDataMat.at<float>(i, j) = featureTrainingData.at(i).at(j);
                            }

                        Mat trainingLabelsMat = Mat(numberToTrain,1,  CV_32SC1);
                        memcpy(trainingLabelsMat.data, binaryTrainingData.data(), binaryTrainingData.size()*sizeof(int));

                        Mat evaluationDataMat = Mat(evaluation_items, numberOfFeatures, CV_32FC1);

                        for(int i=0; i<evaluation_items; ++i)
                            for(int j=0; j<numberOfFeatures; ++j)
                            {
                                evaluationDataMat.at<float>(i, j) = featureEvaluationData.at(i).at(j);
                            }

                        Mat evaluationLabelsMat = Mat(evaluation_items,1,  CV_32SC1);
                        memcpy(evaluationLabelsMat.data, binaryEvaluationData.data(), binaryEvaluationData.size()*sizeof(int));

                        /// Train the SVM ///////////////////////////////////////////////////
                        svm->setC(c);
                        svm->setGamma(g);
                        svm->train(trainingDataMat, ROW_SAMPLE, trainingLabelsMat);

                        int linhas = 0;
                        int ones = 0;
                        int zeros = 0;
                        int acertos = 0;
                        int tp = 0;
                        int tn = 0;
                        int fn = 0;
                        int fp = 0;

                        ///Show the decision given by the SVM /////////////////////////////////////
                        while(linhas < evaluation_items)
                        {
                            Mat newSampleMat(1, numberOfFeatures, CV_32FC1);
                            for(int j=0; j<numberOfFeatures; ++j)
                            {
                                newSampleMat.at<float>(0, j) = featureEvaluationData.at(linhas).at(j);
                            }

                            float response = svm->predict(newSampleMat);

                            ///Compute statistics //////////////////////////////////////////////////
                            if (response == binaryEvaluationData[linhas])
                            {
                                acertos++;
                                if(binaryEvaluationData[linhas] == 1)
                                {
                                    tp++;
                                    ones++;
                                }
                                else
                                {
                                    zeros++;
                                    tn++;
                                }

                            }
                            else
                            {

                                if(binaryEvaluationData[linhas] == 1)
                                {
                                    fn++;
                                    ones++;
                                }
                                else
                                {
                                    zeros++;
                                    fp++;
                                }

                            }
                            linhas++;

                        }

                        double acertoF = ((float)acertos/linhas);
                        double precision = (float)tp/(tp+fp);
                        if(isnan(precision)) precision = 0.0;
                        double recall = (float)tp/(fn+tp);
                        if(isnan(recall)) recall = 0.0;
                        double fpr = (float)fp/(tn+fp);
                        if(isnan(fpr)) fpr = 0.0;
                        double fnr = (float)tn/(tn+fp);

                        crossAccuracy += acertoF;
                        crossPrecision += precision;
                        crossRecall += recall;
                        crossFPR += fpr;
                        crossFNR += fnr;

                        linhas = 0;
                        ones = 0;
                        zeros = 0;
                        acertos = 0;
                        tp = 0;
                        tn = 0;
                        fn = 0;
                        fp = 0;

                        evaluation_start+=evaluation_items;
                        evaluation_end+=evaluation_items;
                        folds++;

                    }

                    if(crossAccuracy/folds > opt_acertos)
                    {
                        opt_acertos = crossAccuracy/folds;
                        opt_c = c;
                        opt_g = g;

                        double optPrecision = crossPrecision/folds;
                        double optRecall =  crossRecall/folds;
                        double F1 = 2*((optPrecision*optRecall)/(optPrecision+optRecall));

                        cout<<"new-> A: "<<opt_acertos<<" "<<" P: "<< crossPrecision/folds
                            <<" R: "<<(double)crossRecall/folds<<" F1: "<< F1<<" C: "<<opt_c<<" G: "<<opt_g<<endl;
                    }

                    evaluation_start = 0;
                    evaluation_end = evaluation_items;
                    g = g*multiplier;

                }
                g = gExpFactor;
                c = c*multiplier;

            }//end of while c g

        }
        else    //end if (!automatic)
        {

            double crossAccuracy = 0.0;
            int folds = 0;

            while(evaluation_start < total_items) //cycle n folds
            {

                loadBinData(evaluation_start, evaluation_end );
                loadData(evaluation_start, evaluation_end );

                ///Setup the input matrixes for training and evaluating of the svm /////////////
                Mat trainingDataMat = Mat(numberToTrain, numberOfFeatures, CV_32FC1);

                for(int i=0; i<numberToTrain; ++i)
                    for(int j=0; j<numberOfFeatures; ++j)
                    {
                        trainingDataMat.at<float>(i, j) = featureTrainingData.at(i).at(j);
                    }
                Mat trainingLabelsMat = Mat(numberToTrain,1,  CV_32SC1);
                memcpy(trainingLabelsMat.data, binaryTrainingData.data(), binaryTrainingData.size()*sizeof(int));

                Mat evaluationDataMat = Mat(evaluation_items, numberOfFeatures, CV_32FC1);

                for(int i=0; i<evaluation_items; ++i)
                    for(int j=0; j<numberOfFeatures; ++j)
                    {
                        evaluationDataMat.at<float>(i, j) = featureEvaluationData.at(i).at(j);
                    }
                Mat evaluationLabelsMat = Mat(evaluation_items,1,  CV_32SC1);
                memcpy(evaluationLabelsMat.data, binaryEvaluationData.data(), binaryEvaluationData.size()*sizeof(int));

                /// Train the SVM ///////////////////////////////////////////////////
                Ptr<TrainData> td = TrainData::create(trainingDataMat, ROW_SAMPLE, trainingLabelsMat);
                svm->trainAuto(td, 10);

                int linhas = 0;
                int ones = 0;
                int zeros = 0;
                int acertos = 0;
                int tp = 0;
                int tn = 0;
                int fn = 0;
                int fp = 0;

                ///Show the decision given by the SVM /////////////////////////////////////
                while(linhas < evaluation_items)
                {
                    Mat newSampleMat(1, numberOfFeatures, CV_32FC1);
                    for(int j=0; j<numberOfFeatures; ++j)
                    {
                        newSampleMat.at<float>(0, j) = featureEvaluationData.at(linhas).at(j);
                    }

                    float response = svm->predict(newSampleMat);

                    ///Compute statistics /////////////////////////////////////////////////////
                    if (response == binaryEvaluationData[linhas])
                    {
                        acertos++;
                        if(binaryEvaluationData[linhas] == 1)
                        {
                            tp++;
                            ones++;
                        }
                        else
                        {
                            zeros++;
                            tn++;
                        }

                    }
                    else
                    {

                        if(binaryEvaluationData[linhas] == 1)
                        {
                            fn++;
                            ones++;
                        }
                        else
                        {
                            zeros++;
                            fp++;
                        }

                    }
                    linhas++;

                }

                float acertoF = ((float)acertos/linhas)*100;
                float precision = (float)tp/(tp+fp);
                if(isnan(precision)) precision = 0.0;
                float recall = (float)tp/(fn+tp);
                float fpr = (float)fp/(tn+fp);
                float fnr = (float)tn/(tn+fp);

                crossAccuracy+=acertoF;

                cout<<" A: "<<acertoF<<"%"<<" P: "<<precision<<" R: "<<recall<<" fnr "<<fnr<<endl;
                cout <<" fp "<<fp<<" tp "<<tp <<" tn "<<tn<<" fn "<<fn<<" O: "<<ones<<" E: "<<linhas<<endl;

                evaluation_start += evaluation_items;
                evaluation_end += evaluation_items;
                folds++;
                linhas = 0;
                ones = 0;
                zeros = 0;
                acertos = 0;
                tp = 0;
                tn = 0;
                fn = 0;
                fp = 0;

            }

            cout<<" cross accuracy: " <<(crossAccuracy/folds)<<"% "<<
                " C: "<<svm->getC()<<" G: "<<svm->getGamma()<<endl;
        }

    }
    catch (const exception &e)
    {
        cout << "error: " << e.what() << endl;
        return -1;
    }

    auto fim = chrono::high_resolution_clock::now();
	cout << "processed in: " << std::chrono::duration_cast<chrono::milliseconds>(fim - start).count()
	<< " ms" << endl;

    return 0;

}

void countData()
{

/// Count features and samples //////////////////////////////////////////////////////
    ifstream inputFeatureCountData(feature_vector);
    string current_line;
    int sampleCount = 0;     //number of samples
    int featureCount = 0;    //number of features

    // Start reading lines as long as there are lines in the file
    while(getline(inputFeatureCountData, current_line))
    {
        stringstream temp(current_line);
        string single_value;

        if (sampleCount==0)
        {
            while(getline(temp,single_value,','))
                featureCount++;
        }
        sampleCount++;
    }

    inputFeatureCountData.close();
    total_items = sampleCount;
    numberOfFeatures = featureCount;

}

void loadData(int ev_start, int ev_end)
{

    /// Set up features data //////////////////////////////////////////////////////
    ifstream inputFeatureData(feature_vector);
    string current_training_line;

    int y = 0;     //sample index
    int tf = 0;    //training sample index
    int ef = 0;    //evaluating sample index

    // Start reading lines as long as there are lines in the file
    while(getline(inputFeatureData, current_training_line))
    {
        stringstream temp(current_training_line);
        string single_value;
        int z = 0;
        vector< double > tempVector(numberOfFeatures,0);
        while(getline(temp,single_value,','))
        {
            tempVector[z] = atof(single_value.c_str()); // convert string to a float
            z++;
        }

        if((y < ev_start || y >= ev_end))
        {
            featureTrainingData[tf] = tempVector;
            tf++;
        }
        else
        {
            featureEvaluationData[ef] = tempVector;
            ef++;
        }
        y++;
    }
    inputFeatureData.close();

}

void loadBinData(int ev_start, int ev_end )
{
    /// Set up label data ///////////////////////////////////////////////////////////
    ifstream inputBinaryData(binary_scores);
    string current_binary_line;

    ev_end = ev_start + evaluation_items;

    int x = 0; //labels index
    int t = 0; //training labels index
    int e = 0; //evaluating labels index

    while(getline(inputBinaryData, current_binary_line))
    {
        int temp = atof(current_binary_line.c_str());

        if(x < ev_start || x >= ev_end)
        {
            binaryTrainingData[t] = temp;
            t++;
        }
        else
        {
            binaryEvaluationData[e] = temp;
            e++;
        }
        x++;
    }
    inputBinaryData.close();
}
string printHelp()
{

    string h ="Help\n"
              "Usage: tool-svm <featureFile> <binaryFile> [arguments]\n\n"
              "Arguments:\n"
              "  -e, number of evaluation samples(default=10): -e=100\n"
              "      Set number of items to 100\n"
              "  -a, automatic optimization(default=0): -a=true\n"
              "      automatic search for optimal C and Gamma ON .\n"
              "  -h, this help message.\n\n";


    return h;
}
