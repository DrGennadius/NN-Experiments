using NNExperiments.Common;
using NNExperiments.Common.Training;
using NNExperiments.Perceptrons.Common;
using System;

namespace NNExperiments.Perceptrons
{
    /// <summary>
    /// Trainer for Perceptrons
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
        /// <param name="printError">Print error each epoch.</param>
        /// <returns></returns>
        public TrainStats Train(IBasicPerceptron perceptron, TrainData trainData, double alpha, double targetError, int maxEpoch, bool printError, int printErrorSteps = -1)
        {
            double error = double.MaxValue;
            int rowCountX = trainData.Inputs.GetLength(0);
            int columnCountX = trainData.Inputs.GetLength(1);
            int rowCountY = trainData.Outputs.GetLength(0);
            int columnCountY = trainData.Outputs.GetLength(1);
            int printErrorStep = printErrorSteps > 0 ? maxEpoch / printErrorSteps : maxEpoch >= 100 ? maxEpoch / 100 : maxEpoch > 10 ? maxEpoch / 10 : 1;
            double[,] outputs = new double[rowCountY, columnCountY];
            int epoch = 0;
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
                if (printError && epoch % printErrorStep == 0)
                {
                    Console.WriteLine("({0}) Mean Batch MSE: {1}", epoch, error);
                }
            }
            while (error > targetError && epoch < maxEpoch);
            TrainStats trainStats = new TrainStats
            {
                LastError = error,
                NumberOfEpoch = epoch
            };
            return trainStats;
        }

        /// <summary>
        /// Pair training of two Perceptrons.
        /// </summary>
        /// <param name="perceptron1">Perceptron 1</param>
        /// <param name="perceptron2">Perceptron 1</param>
        /// <param name="trainData">Data for training.</param>
        /// <param name="alpha">Learning rate.</param>
        /// <param name="targetError">Target error value.</param>
        /// <param name="maxEpoch">Max number epochs.</param>
        /// <param name="printError">Print error each epoch.</param>
        /// <returns></returns>
        public TrainStats[] PairTrain(IBasicPerceptron perceptron1, IBasicPerceptron perceptron2, TrainData trainData, double alpha, double targetError, int maxEpoch, bool printError, int printErrorSteps = -1)
        {
            double error1 = double.MaxValue;
            double error2 = double.MaxValue;
            int rowCountX = trainData.Inputs.GetLength(0);
            int columnCountX = trainData.Inputs.GetLength(1);
            int rowCountY = trainData.Outputs.GetLength(0);
            int columnCountY = trainData.Outputs.GetLength(1);
            int printErrorStep = printErrorSteps > 0 ? maxEpoch / printErrorSteps : maxEpoch >= 100 ? maxEpoch / 100 : maxEpoch > 10 ? maxEpoch / 10 : 1;
            double[,] outputs1 = new double[rowCountY, columnCountY];
            double[,] outputs2 = new double[rowCountY, columnCountY];
            int epoch = 0;
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
                        outputs2[s, c] = currentOutput1[c];
                    }
                    perceptron1.Backward(targetOutputs, alpha);
                    perceptron2.Backward(targetOutputs, alpha);
                }
                error1 = CommonFunctions.MeanBatchMSE(outputs1, trainData.Outputs);
                error2 = CommonFunctions.MeanBatchMSE(outputs2, trainData.Outputs);
                if (printError && epoch % printErrorStep == 0)
                {
                    Console.WriteLine("P1({0}) Mean Batch MSE: {1}", epoch, error1);
                    Console.WriteLine("P2({0}) Mean Batch MSE: {1}", epoch, error2);
                }
            }
            while (error1 > targetError && error2 > targetError && epoch < maxEpoch);
            TrainStats[] trainStats = new TrainStats[2];
            trainStats[0].LastError = error1;
            trainStats[0].NumberOfEpoch = epoch;
            trainStats[1].LastError = error2;
            trainStats[1].NumberOfEpoch = epoch;
            return trainStats;
        }
    }
}
