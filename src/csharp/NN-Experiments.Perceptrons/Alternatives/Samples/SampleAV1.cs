/****************************************************************************************
 * Based on https://programforyou.ru/poleznoe/pishem-neuroset-pryamogo-rasprostraneniya *
 ****************************************************************************************/

using NNExperiments.Common;
using NNExperiments.Common.Training;
using NNExperiments.Perceptrons.Alternatives.V1;
using System;
using System.Text;

namespace NNExperiments.Perceptrons.Alternatives.Samples
{
    /// <summary>
    /// The sample based on programforyou.ru sample.
    /// </summary>
    public class SampleAV1
    {
        /// <summary>
        /// XOR
        /// </summary>
        /// <returns></returns>
        public static string RunXOR()
        {
            StringBuilder result = new();

            // array of input training vectors
            VectorAV1[] X = {
                new VectorAV1(0, 0),
                new VectorAV1(0, 1),
                new VectorAV1(1, 0),
                new VectorAV1(1, 1)
            };

            // array of output training vectors
            VectorAV1[] Y = {
                new VectorAV1(0.0), // 0 ^ 0 = 0
                new VectorAV1(1.0), // 0 ^ 1 = 1
                new VectorAV1(1.0), // 1 ^ 0 = 1
                new VectorAV1(0.0)  // 1 ^ 1 = 0
            };

            PerceptronAV1 perceptron = new(new int[] { 2, 3, 1 });

            perceptron.Train(X, Y, 0.5, 1e-7, 100000);

            for (int i = 0; i < 4; i++)
            {
                VectorAV1 output = perceptron.Forward(X[i]);
                result.Append(string.Format("{0}X: {1} {2}, Y: {3}, output: {4}", i == 0 ? "" : "\n", X[i][0], X[i][1], Y[i][0], output[0]));
            }

            return result.ToString();
        }

        /// <summary>
        /// AND
        /// </summary>
        /// <returns></returns>
        public static string RunAND()
        {
            StringBuilder result = new();

            // array of input training vectors
            VectorAV1[] X = {
                new VectorAV1(0, 0),
                new VectorAV1(0, 1),
                new VectorAV1(1, 0),
                new VectorAV1(1, 1)
            };

            // array of output training vectors
            VectorAV1[] Y = {
                new VectorAV1(0.0), // 0 and 0 = 0
                new VectorAV1(0.0), // 0 and 1 = 0
                new VectorAV1(0.0), // 1 and 0 = 0
                new VectorAV1(1.0)  // 1 and 1 = 1
            };

            PerceptronAV1 perceptron = new(new int[] { 2, 3, 1 });

            perceptron.Train(X, Y, 0.5, 1e-7, 100000);

            for (int i = 0; i < 4; i++)
            {
                VectorAV1 output = perceptron.Forward(X[i]);
                result.Append(string.Format("{0}X: {1} {2}, Y: {3}, output: {4}", i == 0 ? "" : "\n", X[i][0], X[i][1], Y[i][0], output[0]));
            }

            return result.ToString();
        }

        /// <summary>
        /// NAND
        /// </summary>
        /// <returns></returns>
        public static string RunNAND()
        {
            StringBuilder result = new();

            // array of input training vectors
            VectorAV1[] X = {
                new VectorAV1(0, 0),
                new VectorAV1(0, 1),
                new VectorAV1(1, 0),
                new VectorAV1(1, 1)
            };

            // array of output training vectors
            VectorAV1[] Y = {
                new VectorAV1(1.0), // not (0 and 0) = 1
                new VectorAV1(1.0), // not (0 and 1) = 1
                new VectorAV1(1.0), // not (1 and 0) = 1
                new VectorAV1(0.0)  // not (1 and 1) = 0
            };

            PerceptronAV1 perceptron = new(new int[] { 2, 3, 1 });

            perceptron.Train(X, Y, 0.5, 1e-7, 100000);

            for (int i = 0; i < 4; i++)
            {
                VectorAV1 output = perceptron.Forward(X[i]);
                result.Append(string.Format("{0}X: {1} {2}, Y: {3}, output: {4}", i == 0 ? "" : "\n", X[i][0], X[i][1], Y[i][0], output[0]));
            }

            return result.ToString();
        }

        /// <summary>
        /// OR
        /// </summary>
        /// <returns></returns>
        public static string RunOR()
        {
            StringBuilder result = new();

            // array of input training vectors
            VectorAV1[] X = {
                new VectorAV1(0, 0),
                new VectorAV1(0, 1),
                new VectorAV1(1, 0),
                new VectorAV1(1, 1)
            };

            // array of output training vectors
            VectorAV1[] Y = {
                new VectorAV1(0.0), // 0 and 0 = 0
                new VectorAV1(1.0), // 0 and 1 = 1
                new VectorAV1(1.0), // 1 and 0 = 1
                new VectorAV1(1.0)  // 1 and 1 = 1
            };

            PerceptronAV1 perceptron = new(new int[] { 2, 3, 1 });

            perceptron.Train(X, Y, 0.5, 1e-7, 100000);

            for (int i = 0; i < 4; i++)
            {
                VectorAV1 output = perceptron.Forward(X[i]);
                result.Append(string.Format("{0}X: {1} {2}, Y: {3}, output: {4}", i == 0 ? "" : "\n", X[i][0], X[i][1], Y[i][0], output[0]));
            }

            return result.ToString();
        }

        /// <summary>
        /// Comparison with a simple perceptron
        /// </summary>
        /// <returns></returns>
        public static string Comparison()
        {
            StringBuilder result = new();

            // array of input training vectors
            VectorAV1[] X = {
                new VectorAV1(0, 0),
                new VectorAV1(0, 1),
                new VectorAV1(1, 0),
                new VectorAV1(1, 1)
            };

            // array of output training vectors
            VectorAV1[] Y = {
                new VectorAV1(0.0), // 0 and 0 = 0
                new VectorAV1(1.0), // 0 and 1 = 0
                new VectorAV1(1.0), // 1 and 0 = 0
                new VectorAV1(0.0)  // 1 and 1 = 1
            };

            TrainData trainData = TrainData.GenerateDataXOR();

            Random random = new(134577);
            SimplePerceptron perceptron1 = new(new int[] { 2, 3, 1 }, random);
            PerceptronAV1 perceptron2 = new(perceptron1);

            TrainStats[] stats = PairComparisonTrain(perceptron1, trainData, perceptron2, new VectorAV1[][] { X, Y }, 1e-7, 0.5, 1000000, true);

            result.Append("\nSimplePerceptron:\n");
            result.Append(string.Format("\tLast error: {0:0.0000000};  Epoch: {1}",
                stats[0].LastError, stats[0].NumberOfEpoch));
            for (int i = 0; i < 4; i++)
            {
                double[] input = new double[2];
                for (int k = 0; k < 2; k++)
                {
                    input[k] = trainData.Inputs[i, k];
                }
                double[] output = perceptron1.Forward(input);
                result.Append(string.Format("\n\tX: {0} {1}, Y: {2}, output: {3}", X[i][0], X[i][1], Y[i][0], output[0]));
            }

            result.Append("\n\nPerceptronAV1:\n");
            result.Append(string.Format("\tLast error: {0:0.0000000};  Epoch: {1}",
                stats[1].LastError, stats[1].NumberOfEpoch));
            for (int i = 0; i < 4; i++)
            {
                VectorAV1 output = perceptron2.Forward(X[i]);
                result.Append(string.Format("\n\tX: {0} {1}, Y: {2}, output: {3}", X[i][0], X[i][1], Y[i][0], output[0]));
            }

            return result.ToString();
        }

        public static TrainStats[] PairComparisonTrain(IPerceptronOld perceptron1, TrainData trainData, PerceptronAV1 perceptron2, VectorAV1[][] trainDataAV1, double targetError, double learningRate, int maxEpoch, bool printError)
        {
            double error1 = double.MaxValue;
            double error2 = double.MaxValue;
            int rowCountX = trainData.Inputs.GetLength(0);
            int columnCountX = trainData.Inputs.GetLength(1);
            int rowCountY = trainData.Outputs.GetLength(0);
            int columnCountY = trainData.Outputs.GetLength(1);
            int printErrorStep = maxEpoch >= 100 ? maxEpoch / 100 : maxEpoch > 10 ? maxEpoch / 10 : 1;
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
                    var currentOutput2 = perceptron2.Forward(trainDataAV1[0][s]);
                    for (var c = 0; c < columnCountY; c++)
                    {
                        outputs1[s, c] = currentOutput1[c];
                        outputs2[s, c] = currentOutput1[c];
                    }
                    perceptron1.Backward(input, targetOutputs, learningRate);
                    double tempError = 0;
                    perceptron2.Backward(trainDataAV1[1][s], ref tempError);
                    perceptron2.UpdateWeights(learningRate);
                }
                error1 = CommonFunctions.MeanBatchMSE(outputs1, trainData.Outputs);
                error2 = CommonFunctions.MeanBatchMSE(outputs2, trainData.Outputs);
                if (printError && epoch % printErrorStep == 0)
                {
                    Console.WriteLine("P1({0}) Mean batch MSE: {1}", epoch, error1);
                    Console.WriteLine("P2({0}) Mean batch MSE: {1}", epoch, error2);
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
