using NNExperiments.Common.ActivationFunctions;
using System;

namespace NNExperiments.Perceptrons.Alternatives
{
    public class LayerOld
    {
        public NeuronOld[] Neurons;

        public LayerOld(double[][] layerWeights, double[] layerBias)
        {
            int neuronCount = layerWeights.GetLength(0);
            Neurons = new NeuronOld[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new NeuronOld(layerWeights[i], layerBias[i]);
            }
        }

        public LayerOld(double[][] layerWeights, Random r)
        {
            int neuronCount = layerWeights.GetLength(0);
            Neurons = new NeuronOld[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new NeuronOld(layerWeights[i], r);
            }
        }

        public LayerOld(int numberOfNeurons, int numberOfInputs, Random r)
        {
            Neurons = new NeuronOld[numberOfNeurons];
            for (int i = 0; i < numberOfNeurons; i++)
            {
                Neurons[i] = new NeuronOld(numberOfInputs, r);
            }
        }

        public double[] FeedForward(double[] input)
        {
            double[] output = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                output[i] = Neurons[i].FeedForward(input);
            }
            return output;
        }

        public double[] FeedForward(ActivationFunction activationFunction, double[] input)
        {
            double[] output = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                output[i] = Neurons[i].FeedForward(activationFunction, input);
            }
            return output;
        }
    }
}
