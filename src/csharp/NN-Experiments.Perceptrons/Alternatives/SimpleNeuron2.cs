using NNExperiments.Common.ActivationFunctions;
using System;

namespace NNExperiments.Perceptrons.Alternatives
{
    public class SimpleNeuron2
    {
        public double[] Weights;
        public double[] Input;
        public double Output;
        public double DerivatedOutput;
        public double Bias;

        public SimpleNeuron2(double[] neuronWeights, Random r)
        {
            Bias = r.NextDouble() - 0.5;
            Weights = neuronWeights;
        }

        public SimpleNeuron2(int numberOfInputs, Random r)
        {
            Bias = 2 * r.NextDouble() - 1;
            Weights = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                Weights[i] = r.NextDouble() - 0.5;
            }
        }

        public double FeedForward(double[] input)
        {
            Input = input;

            double sum = Bias;

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
