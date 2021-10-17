using System;

namespace NNExperiments.Perceptrons.Alternatives
{
    public class SimpleLayer2
    {
        public SimpleNeuron2[] Neurons;

        public SimpleLayer2(double[][] layerWeights, Random r)
        {
            int neuronCount = layerWeights.GetLength(0);
            Neurons = new SimpleNeuron2[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new SimpleNeuron2(layerWeights[i], r);
            }
        }

        public SimpleLayer2(int numberOfNeurons, int numberOfInputs, Random r)
        {
            Neurons = new SimpleNeuron2[numberOfNeurons];
            for (int i = 0; i < numberOfNeurons; i++)
            {
                Neurons[i] = new SimpleNeuron2(numberOfInputs, r);
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
