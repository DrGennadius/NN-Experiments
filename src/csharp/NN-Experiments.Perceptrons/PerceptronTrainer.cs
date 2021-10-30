using NNExperiments.Common;
using NNExperiments.Common.Training;
using NNExperiments.Perceptrons.Common;
using System;
using System.Diagnostics;

namespace NNExperiments.Perceptrons
{
    /// <summary>
    /// Perceptron Trainer. Performs training of perceptron models.
    /// </summary>
    public class PerceptronTrainer
    {
        /// <summary>
        /// Training Perceptron.
        /// </summary>
        /// <param name="perceptron">Perceptron.</param>
        /// <param name="trainData">Data for training.</param>
        /// <param name="alpha">Learning rate.</param>
        /// <param name="targetError">Target error value.</param>
        /// <param name="maxEpoch">Max number epochs.</param>
        /// <param name="isStoreErrorHistory">Is store error history?</param>
        /// <param name="storeErrorSteps">Steps for store an errors.</param>
        /// <returns></returns>
        public TrainStats Train(IBasicPerceptron perceptron,
                                TrainData trainData,
                                double alpha,
                                double targetError,
                                int maxEpoch,
                                bool isStoreErrorHistory,
                                int storeErrorSteps = -1)
        {
            TrainStats trainStats = new();
            double error = double.MaxValue;
            int rowCountX = trainData.Inputs.GetLength(0);
            int columnCountX = trainData.Inputs.GetLength(1);
            int rowCountY = trainData.Outputs.GetLength(0);
            int columnCountY = trainData.Outputs.GetLength(1);
            int errorHistoryStep = CalcErrorHistoryStep(maxEpoch, storeErrorSteps);
            double[,] outputs = new double[rowCountY, columnCountY];
            int epoch = 0;
            var stopwatch = Stopwatch.StartNew();
            do
            {
                epoch++;
                for (int s = 0; s < trainData.Inputs.GetLength(0); s++)
                {
                    double[] input = new double[columnCountX];
                    double[] targetOutputs = new double[columnCountY];
                    for (var c = 0; c < columnCountX; c++)
                    {
                        input[c] = trainData.Inputs[s, c];
                    }
                    for (var c = 0; c < columnCountY; c++)
                    {
                        targetOutputs[c] = trainData.Outputs[s, c];
                    }
                    var currentOutput = perceptron.Forward(input);
                    for (var c = 0; c < columnCountY; c++)
                    {
                        outputs[s, c] = currentOutput[c];
                    }
                    perceptron.Backward(targetOutputs, alpha);
                }
                error = CommonFunctions.MeanBatchMSE(outputs, trainData.Outputs);
                if (isStoreErrorHistory && epoch % errorHistoryStep == 0)
                {
                    trainStats.ErrorHistory.Add(new EpochStats(epoch, error));
                }
            }
            while (error > targetError
                   && epoch < maxEpoch);

            stopwatch.Stop();

            trainStats.Time = new TimeSpan(stopwatch.ElapsedTicks);
            trainStats.LastError = error;
            trainStats.NumberOfEpoch = epoch;

            return trainStats;
        }

        /// <summary>
        /// Pair training of the two perceptrons.
        /// </summary>
        /// <param name="perceptron1">Perceptron 1</param>
        /// <param name="perceptron2">Perceptron 1</param>
        /// <param name="trainData">Data for training.</param>
        /// <param name="alpha">Learning rate.</param>
        /// <param name="targetError">Target error value.</param>
        /// <param name="maxEpoch">Max number epochs.</param>
        /// <param name="isStoreErrorHistory">Is store error history?</param>
        /// <param name="storeErrorSteps">Steps for store an errors.</param>
        /// <returns></returns>
        public TrainStats[] PairTrain(IBasicPerceptron perceptron1,
                                      IBasicPerceptron perceptron2,
                                      TrainData trainData,
                                      double alpha,
                                      double targetError,
                                      int maxEpoch,
                                      bool isStoreErrorHistory,
                                      int storeErrorSteps = -1)
        {
            TrainStats[] trainStats = new TrainStats[2];
            int rowCountX = trainData.Inputs.GetLength(0);
            int columnCountX = trainData.Inputs.GetLength(1);
            int rowCountY = trainData.Outputs.GetLength(0);
            int columnCountY = trainData.Outputs.GetLength(1);
            int errorHistoryStep = CalcErrorHistoryStep(maxEpoch, storeErrorSteps);
            double[,] outputs1 = new double[rowCountY, columnCountY];
            double[,] outputs2 = new double[rowCountY, columnCountY];
            int epoch = 0;
            double error1;
            double error2;
            do
            {
                epoch++;
                for (int s = 0; s < trainData.Inputs.GetLength(0); s++)
                {
                    double[] input = new double[columnCountX];
                    double[] targetOutputs = new double[columnCountY];
                    for (var c = 0; c < columnCountX; c++)
                    {
                        input[c] = trainData.Inputs[s, c];
                    }
                    for (var c = 0; c < columnCountY; c++)
                    {
                        targetOutputs[c] = trainData.Outputs[s, c];
                    }
                    var currentOutput1 = perceptron1.Forward(input);
                    var currentOutput2 = perceptron2.Forward(input);
                    for (var c = 0; c < columnCountY; c++)
                    {
                        outputs1[s, c] = currentOutput1[c];
                        outputs2[s, c] = currentOutput2[c];
                    }
                    perceptron1.Backward(targetOutputs, alpha);
                    perceptron2.Backward(targetOutputs, alpha);
                }
                error1 = CommonFunctions.MeanBatchMSE(outputs1, trainData.Outputs);
                error2 = CommonFunctions.MeanBatchMSE(outputs2, trainData.Outputs);
                if (isStoreErrorHistory && epoch % errorHistoryStep == 0)
                {
                    trainStats[0].ErrorHistory.Add(new EpochStats(epoch, error1));
                    trainStats[1].ErrorHistory.Add(new EpochStats(epoch, error2));
                }
            }
            while (error1 > targetError
                   && error2 > targetError
                   && epoch < maxEpoch);

            trainStats[0].LastError = error1;
            trainStats[0].NumberOfEpoch = epoch;
            trainStats[1].LastError = error2;
            trainStats[1].NumberOfEpoch = epoch;

            return trainStats;
        }

        /// <summary>
        /// Calculate the step of the error history. Each step will store the error value.
        /// </summary>
        /// <param name="maxEpoch"></param>
        /// <param name="storeErrorSteps"></param>
        /// <returns></returns>
        private static int CalcErrorHistoryStep(int maxEpoch, int storeErrorSteps)
        {
            if (storeErrorSteps > 0)
            {
                return maxEpoch / storeErrorSteps;
            }
            return maxEpoch switch
            {
                >= 100 => maxEpoch / 100,
                > 10 => maxEpoch / 10,
                _ => 1,
            };
        }
    }
}
