using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static void Main(string[] args)
        {
            MLContext mLContext = new MLContext();
            TrainTestData splitDataView = LoadData(mLContext);
            ITransformer model = BuildAndTrainModel(mLContext, splitDataView.TrainSet);
            Evaluate(mLContext, model, splitDataView.TestSet);
            UseModelWithSingleItem(mLContext, model);
        }
        public static TrainTestData LoadData(MLContext mLContext)
        {
            IDataView dataView = mLContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = mLContext.Data.TrainTestSplit(data: dataView, testFraction: 0.2);
            return splitDataView;
        }

        public static ITransformer BuildAndTrainModel(MLContext mLContext, IDataView splitTrainSet)
        {
            var estimator = mLContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mLContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            Console.WriteLine("======== CREATE AND TRAIN THE MODEL");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("======== END OF TRAINING");
            Console.WriteLine();
            return model;
        }

        public static void Evaluate(MLContext mLContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("======== EVALUATING");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };
            var resultprediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");
            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultprediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultprediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }
    }
}
