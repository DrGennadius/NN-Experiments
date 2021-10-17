using System;

namespace NNExperiments.Perceptrons.Alternatives
{
    public class SimpleLayer
    {
        public SimpleNeuron[] Neurons;

        public SimpleLayer(double[][] layerWeights)
        {
            int neuronCount = layerWeights.GetLength(0);
            Neurons = new SimpleNeuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new SimpleNeuron(layerWeights[i]);
            }
        }

        public SimpleLayer(int numberOfNeurons, int numberOfInputs, Random r)
        {
            Neurons = new SimpleNeuron[numberOfNeurons];
            for (int i = 0; i < numberOfNeurons; i++)
            {
                Neurons[i] = new SimpleNeuron(numberOfInputs, r);
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
    }
}
