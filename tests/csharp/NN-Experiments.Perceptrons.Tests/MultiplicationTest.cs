using NNExperiments.Common;
using NNExperiments.Common.ActivationFunctions;
using NNExperiments.Common.Training;
using NNExperiments.Perceptrons.Common;
using NUnit.Framework;
using System;

namespace NNExperiments.Perceptrons.Tests
{
    public class MultiplicationTest
    {
        private const int Seed = 2387648;
        private readonly int[] _netConfig = new int[] { 2, 5, 3, 1 };

        private const double LearningRate = 0.1;
        private const double TargetError = 1e-7;
        private const int SmallBatchMaxEpoch = 250000;
        private const int BigBatchMaxIterations = 100;
        private const bool IsStoreErrorHistory = true;

        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void SequentialMultiplicationTest()
        {
            var multiplicationData = TrainData.GenerateDataSequentialMultiplication();
            var normalizedMultiplicationData = CommonFunctions.NormalizeTrainData(multiplicationData);
            PerceptronTopology topology = new(_netConfig, new ActivationFunction(ActivationFunctionType.Sigmoid));
            Perceptron perceptron = new(topology);

            var perceptronTrainer = new PerceptronTrainer();
            var trainStats = perceptronTrainer.Train(perceptron,
                                                     normalizedMultiplicationData,
                                                     LearningRate,
                                                     TargetError,
                                                     SmallBatchMaxEpoch,
                                                     IsStoreErrorHistory);
            Console.WriteLine(trainStats);

            double[] outputArray = perceptron.Forward(CommonFunctions.Scale(new double[] { 6, 6 }, 0, 9, 0, 1));
            double output = CommonFunctions.Scale(outputArray[0], 0, 1, 0, 9 * 9);
            Assert.AreEqual((int)Math.Round(output), 6 * 6);

            Assert.Pass();
        }

        [Test]
        public void RandomMultiplicationTest()
        {
            Random random = new(Seed);

            PerceptronTopology topology = new(_netConfig, new ActivationFunction(ActivationFunctionType.RandomLReLU));
            Perceptron perceptron = new(topology);

            var perceptronTrainer = new PerceptronTrainer();
            double[] trainErrors = new double[100];
            for (int i = 0; i < BigBatchMaxIterations; i++)
            {
                var multiplicationData = TrainData.GenerateDataRandomMultiplication(1000000, random.Next(), 0, 999);
                multiplicationData = CommonFunctions.NormalizeTrainData(multiplicationData);
                var trainStats = perceptronTrainer.Train(perceptron,
                                                         multiplicationData,
                                                         LearningRate,
                                                         TargetError,
                                                         1,
                                                         IsStoreErrorHistory);
                Console.WriteLine(trainStats);
                trainErrors[i] = trainStats.LastError;
            }

            double[][] testInputs = new double[][]
            {
                new double[] { 0, 0 },
                new double[] { 999, 999 },
                new double[] { 555, 555 },
                new double[] { 2, 100 },
                new double[] { 100, 800 },
                new double[] { 5, 12 },
                new double[] { 123, 987 },
                new double[] { 777, 69 },
                new double[] { 999, 0 },
                new double[] { 0, 999 },
                new double[] { 3, 500 },
                new double[] { 555, 111 },
                new double[] { 111, 555 },
                new double[] { 777, 777 },
                new double[] { 50, 90 },
                new double[] { 33, 333 },
                new double[] { 256, 256 },
                new double[] { 321, 123 },
                new double[] { 123, 321 },
                new double[] { 888, 666 }
            };
            double error = 0;
            for (int i = 0; i < testInputs.Length; i++)
            {
                double x1 = testInputs[i][0];
                double x2 = testInputs[i][1];
                double[] outputArray = perceptron.Forward(CommonFunctions.Scale(new double[] { x1, x2 },
                                                                                new Range<double>(0, 999),
                                                                                new Range<double>(0, 1)));
                double y = CommonFunctions.Scale(outputArray[0],
                                                 new Range<double>(0, 1),
                                                 new Range<double>(0, 999 * 999));
                double e = x1 * x2 - y;
                error += Math.Abs(e);
            }
            error /= testInputs.Length;

            Assert.Less(error, 10000);

            Assert.Pass();
        }
    }
}
