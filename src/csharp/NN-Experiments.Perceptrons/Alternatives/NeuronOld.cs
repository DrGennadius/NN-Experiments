using NNExperiments.Common.ActivationFunctions;
using System;

namespace NNExperiments.Perceptrons.Alternatives
{
    /// <summary>
    /// Neuron for perceptron.
    /// </summary>
    public class NeuronOld
    {
        public double[] Weights;
        public double[] Input;
        public double[] PreviousChanges;
        public double Output;
        public double DerivatedOutput;
        public double Bias;
        public double PreviousBiasChange;

        public NeuronOld(double[] neuronWeights, double bias)
        {
            Bias = bias;
            Weights = neuronWeights;
            PreviousChanges = new double[neuronWeights.Length];
            for (int i = 0; i < neuronWeights.Length; i++)
            {
                PreviousChanges[i] = 0;
            }
        }

        public NeuronOld(double[] neuronWeights, Random r)
            : this(neuronWeights, r.NextDouble() - 0.5)
        {
        }

        public NeuronOld(int numberOfInputs, Random r)
        {
            Bias = r.NextDouble() - 0.5;
            Weights = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                Weights[i] = r.NextDouble() - 0.5;
            }
            PreviousChanges = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                PreviousChanges[i] = 0;
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

            Output = ActivationFunctions.HyperbolicTangent(sum);
            DerivatedOutput = ActivationFunctions.HyperbolicTangentDerivated(Output);
            return Output;
        }

        public double FeedForward(ActivationFunction activationFunction, double[] input)
        {
            Input = input;

            double sum = Bias;

            for (int i = 0; i < Weights.Length; i++)
            {
                sum += Weights[i] * input[i];
            }

            Output = activationFunction.Calculate(sum);
            DerivatedOutput = activationFunction.CalculateDerivative(Output);
            return Output;
        }
    }
}
