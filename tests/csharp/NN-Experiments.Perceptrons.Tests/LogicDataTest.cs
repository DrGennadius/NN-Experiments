using NNExperiments.Common;
using NNExperiments.Common.ActivationFunctions;
using NNExperiments.Common.Training;
using NNExperiments.Perceptrons.Alternatives;
using NNExperiments.Perceptrons.Common;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NNExperiments.Perceptrons.Tests
{
    public class LogicDataTest
    {
        private const double LearningRate = 0.1;
        private const double TargetError = 1e-6;
        private const int MaxEpoch = 1000000;
        private const bool PrintError = false;

        private readonly int[] _netConfig = new int[] { 2, 5, 1 };

        private readonly Dictionary<string, TrainData> _trainLogicDataDictionary = new()
        {
            { "AND", TrainData.GenerateDataAND() },
            { "NAND", TrainData.GenerateDataNAND() },
            { "OR", TrainData.GenerateDataOR() },
            { "XOR", TrainData.GenerateDataXOR() }
        };

        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void SigmoidTest()
        {
            var perceptrons = CreateAllPerceptrons(_netConfig, ActivationFunctionType.Sigmoid);
            TestWithDifferentData(perceptrons);

            Assert.Pass();
        }

        [Test]
        public void TanhTest()
        {
            var perceptrons = CreateAllPerceptrons(_netConfig, ActivationFunctionType.Tanh);
            TestWithDifferentData(perceptrons);

            Assert.Pass();
        }

        [Test]
        public void ReLUTest()
        {
            var perceptrons = CreateAllPerceptrons(_netConfig, ActivationFunctionType.ReLU);
            TestWithDifferentData(perceptrons);

            Assert.Pass();
        }

        [Test]
        public void LReLUTest()
        {
            var perceptrons = CreateAllPerceptrons(_netConfig, ActivationFunctionType.LReLU);
            TestWithDifferentData(perceptrons);

            Assert.Pass();
        }

        [Test]
        public void RandomLReLUTest()
        {
            var perceptrons = CreateAllPerceptrons(_netConfig, ActivationFunctionType.RandomLReLU);
            TestWithDifferentData(perceptrons);

            Assert.Pass();
        }

        public IPerceptronBase[] CreateAllPerceptrons(int[] netConfig, ActivationFunctionType activationFunctionType)
        {
            PerceptronTopology topology = new(netConfig, new ActivationFunction(activationFunctionType));
            object[] masterArgs = new object[] { topology };
            object[] alternativeArgs = new object[] { netConfig, activationFunctionType };
            var perceptronBaseType = typeof(IPerceptronBase);
            var types = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(s => s.GetTypes())
                .Where(p => perceptronBaseType.IsAssignableFrom(p) && !p.IsInterface);

            List<IPerceptronBase> perceptrons = new();

            foreach (var type in types)
            {
                if (type == typeof(SimplePerceptron)
                    || type == typeof(SimplePerceptron2))
                {
                    continue;
                }

                IPerceptronBase newPerceptron = null;

                try
                {
                    newPerceptron = (IPerceptronBase)Activator.CreateInstance(type, masterArgs);
                }
                catch (MissingMethodException)
                {
                    newPerceptron = (IPerceptronBase)Activator.CreateInstance(type, alternativeArgs);
                }

                if (newPerceptron != null)
                {
                    perceptrons.Add(newPerceptron);
                }
            }

            return perceptrons.ToArray();
        }

        private void TestWithDifferentData(IPerceptronBase[] perceptrons)
        {
            foreach (var perceptron in perceptrons)
            {
                TestWithDifferentData(perceptron);
            }
        }

        private void TestWithDifferentData(IPerceptronBase perceptron)
        {
            foreach (var trainDataItem in _trainLogicDataDictionary)
            {
                var perceptronClone = perceptron.Clone() as IPerceptronBase;
                TrainStats trainStats = Train(perceptronClone, trainDataItem.Value);
                TestReadyLogicModel(perceptronClone, trainDataItem.Value, trainDataItem.Key);
            }
        }

        private TrainStats Train(IPerceptronBase perceptron, TrainData trainData)
        {
            return Train(perceptron, trainData, perceptron.Topology.GetOutputActivationFunction().Type);
        }

        private TrainStats Train(IPerceptronBase perceptron, TrainData trainData, ActivationFunctionType activationFunctionType)
        {
            if (activationFunctionType == ActivationFunctionType.Tanh)
            {
                trainData = CommonFunctions.CreateNewNormalizeTrainData(trainData, -1, 1);
            }

            var perceptronTrainer = new PerceptronTrainer();
            return perceptronTrainer.Train(perceptron, trainData, LearningRate, TargetError, MaxEpoch, PrintError);
        }

        private void TestReadyLogicModel(IPerceptronBase perceptron, TrainData trainData, string trainDataName)
        {
            for (int i = 0; i < 4; i++)
            {
                double[] sample = trainData.GetInputRow(i);
                double expectedValue = trainData.GetOutputRow(i)[0];
                TestReadyLogicModel(perceptron, sample, expectedValue, trainDataName);
            }
        }

        private void TestReadyLogicModel(IPerceptronBase perceptron, double[] sample, double expectedValue, string trainDataName)
        {
            var activationFunctionType = perceptron.Topology.GetOutputActivationFunction().Type;
            if (activationFunctionType == ActivationFunctionType.Tanh)
            {
                sample = CommonFunctions.Scale(sample, 0, 1, -1, 1);
            }
            double predictedValue = perceptron.Forward(sample)[0];
            CheckReadyLogicModelOutput(perceptron, sample, expectedValue, predictedValue, activationFunctionType, trainDataName);
        }

        private void CheckReadyLogicModelOutput(IPerceptronBase perceptron,
                                                double[] sample,
                                                double expectedValue,
                                                double predictedValue,
                                                ActivationFunctionType activationFunctionType,
                                                string trainDataName)
        {
            if (activationFunctionType == ActivationFunctionType.Tanh)
            {
                predictedValue = CommonFunctions.Scale(predictedValue, -1, 1, 0, 1);
            }
            Assert.AreEqual(
                (int)Math.Round(expectedValue), 
                (int)Math.Round(predictedValue),
                $"{perceptron}: [{trainDataName}] The predicted result {predictedValue}"
                + $" does not match the expected result {expectedValue} for the sample: {string.Join(' ', sample)}.");
        }
    }
}
