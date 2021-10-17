using NNExperiments.Common.ActivationFunctions;
using System;

namespace NNExperiments.Perceptrons.Alternatives
{
    public class SimpleNeuron
    {
        public double[] Weights;
        public double[] Input;
        public double Output;
        public double DerivatedOutput;

        public SimpleNeuron(double[] neuronWeights)
        {
            Weights = neuronWeights;
        }

        public SimpleNeuron(int numberOfInputs, Random r)
        {
            Weights = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                Weights[i] = r.NextDouble() - 0.5;
            }
        }

        public double FeedForward(double[] input)
        {
            Input = input;

            double sum = 0.0;

            for (int i = 0; i < Weights.Length; i++)
            {
                sum += Weights[i] * input[i];
            }

            Output = ActivationFunctions.Sigmoid(sum);
            DerivatedOutput = ActivationFunctions.SigmoidDerivated(Output);
            return Output;
        }
    }
}
