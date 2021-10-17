using NNExperiments.Common.ActivationFunctions;
using NNExperiments.Perceptrons.Alternatives;
using NUnit.Framework;
using System;

namespace NNExperiments.Perceptrons.Tests
{
    public class PerceptronActivationFunctionTest
    {
        private readonly int[] _netConfig = new int[] { 2, 5, 1 };

        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void SigmoidTest()
        {
            PerceptronOld perceptronAND = new(_netConfig, ActivationFunctionType.Sigmoid);
            PerceptronOld perceptronNAND = new(perceptronAND, ActivationFunctionType.Sigmoid);
            PerceptronOld perceptronOR = new(perceptronAND, ActivationFunctionType.Sigmoid);
            PerceptronOld perceptronXOR = new(perceptronAND, ActivationFunctionType.Sigmoid);

            // TODO: Just finish it.

            Assert.Pass();
        }

        [Test]
        public void TanhTest()
        {
            PerceptronOld perceptronAND = new(_netConfig, ActivationFunctionType.Tanh);
            PerceptronOld perceptronNAND = new(perceptronAND, ActivationFunctionType.Tanh);
            PerceptronOld perceptronOR = new(perceptronAND, ActivationFunctionType.Tanh);
            PerceptronOld perceptronXOR = new(perceptronAND, ActivationFunctionType.Tanh);

            // TODO: Just finish it.

            Assert.Pass();
        }

        [Test]
        public void ReLUTest()
        {
            PerceptronOld perceptronAND = new(_netConfig, ActivationFunctionType.ReLU);
            PerceptronOld perceptronNAND = new(perceptronAND, ActivationFunctionType.ReLU);
            PerceptronOld perceptronOR = new(perceptronAND, ActivationFunctionType.ReLU);
            PerceptronOld perceptronXOR = new(perceptronAND, ActivationFunctionType.ReLU);

            // TODO: Just finish it.

            Assert.Pass();
        }

        [Test]
        public void LReLUTest()
        {
            PerceptronOld perceptronAND = new(_netConfig, ActivationFunctionType.LReLU);
            PerceptronOld perceptronNAND = new(perceptronAND, ActivationFunctionType.LReLU);
            PerceptronOld perceptronOR = new(perceptronAND, ActivationFunctionType.LReLU);
            PerceptronOld perceptronXOR = new(perceptronAND, ActivationFunctionType.LReLU);

            // TODO: Just finish it.

            Assert.Pass();
        }

        [Test]
        public void RandomLReLUTest()
        {
            PerceptronOld perceptronAND = new(_netConfig, ActivationFunctionType.RandomLReLU);
            PerceptronOld perceptronNAND = new(perceptronAND, ActivationFunctionType.RandomLReLU);
            PerceptronOld perceptronOR = new(perceptronAND, ActivationFunctionType.RandomLReLU);
            PerceptronOld perceptronXOR = new(perceptronAND, ActivationFunctionType.RandomLReLU);

            // TODO: Just finish it.

            Assert.Pass();
        }
    }
}