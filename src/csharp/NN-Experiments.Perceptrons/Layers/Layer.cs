using NNExperiments.Common.ActivationFunctions;
using NNExperiments.Perceptrons.Neurons;
using System;
using System.Linq;

namespace NNExperiments.Perceptrons.Layers
{
    /// <summary>
    /// Perceptron layer.
    /// </summary>
    public class Layer : ILayerBase
    {
        public ActivationFunction ActivationFunction { get; set; }
        public Neuron[] Neurons { get; set; }

        public double[] Output
        {
            get
            {
                return Neurons.Select(x => x.Output).ToArray();
            }
        }

        public double[][] Weights
        {
            get
            {
                double[][] weights = new double[Neurons.Length][];
                for (int n = 0; n < Neurons.Length; n++)
                {
                    weights[n] = new double[Neurons[n].Weights.Length];
                    for (int w = 0; w < Neurons[n].Weights.Length; w++)
                    {
                        weights[n][w] = Neurons[n].Weights[w];
                    }
                }
                return weights;
            }
        }

        public Layer(double[][] layerWeights, double[] layerBias)
            : this(layerWeights, layerBias, new ActivationFunction(ActivationFunctionType.Sigmoid))
        {
        }

        public Layer(double[][] layerWeights, double[] layerBias, ActivationFunction activationFunction)
        {
            int neuronCount = layerWeights.GetLength(0);
            Neurons = new Neuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new Neuron(layerWeights[i], layerBias[i]);
            }
            ActivationFunction = activationFunction;
        }

        public Layer(double[][] layerWeights, Random random)
            : this(layerWeights, new ActivationFunction(ActivationFunctionType.Sigmoid), random)
        {
        }

        public Layer(double[][] layerWeights, ActivationFunction activationFunction, Random random)
        {
            int neuronCount = layerWeights.GetLength(0);
            Neurons = new Neuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new Neuron(layerWeights[i], random);
            }
            ActivationFunction = activationFunction;
        }

        public Layer(double[,] layerWeights, double[] layerBias)
            : this(layerWeights, layerBias, new ActivationFunction(ActivationFunctionType.Sigmoid))
        {
        }

        public Layer(double[,] layerWeights, double[] layerBias, ActivationFunction activationFunction)
        {
            int neuronCount = layerWeights.GetLength(0);
            Neurons = new Neuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                double[] neuronWeights = new double[layerWeights.GetLength(1)];
                for (int w = 0; w < layerWeights.GetLength(1); w++)
                {
                    neuronWeights[w] = layerWeights[i, w];
                }
                Neurons[i] = new Neuron(neuronWeights, layerBias[i]);
            }
            ActivationFunction = activationFunction;
        }

        public Layer(double[,] layerWeights, Random random)
            : this(layerWeights, new ActivationFunction(ActivationFunctionType.Sigmoid), random)
        {
        }

        public Layer(double[,] layerWeights, ActivationFunction activationFunction, Random random)
        {
            int neuronCount = layerWeights.GetLength(0);
            Neurons = new Neuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                double[] neuronWeights = new double[layerWeights.GetLength(1)];
                for (int w = 0; w < layerWeights.GetLength(1); w++)
                {
                    neuronWeights[w] = layerWeights[i, w];
                }
                Neurons[i] = new Neuron(neuronWeights, random);
            }
            ActivationFunction = activationFunction;
        }

        public Layer(int numberOfNeurons, int numberOfInputs, Random random)
            : this(numberOfNeurons, numberOfInputs, new ActivationFunction(ActivationFunctionType.Sigmoid), random)
        {
        }

        public Layer(int numberOfNeurons, int numberOfInputs, ActivationFunction activationFunction, Random random)
        {
            Neurons = new Neuron[numberOfNeurons];
            for (int i = 0; i < numberOfNeurons; i++)
            {
                Neurons[i] = new Neuron(numberOfInputs, random);
            }
            ActivationFunction = activationFunction;
        }

        public Layer(ILayerBase layer)
        {
            Neurons = new Neuron[layer.Neurons.Length];
            for (int i = 0; i < layer.Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(layer.Neurons[i]);
            }
            ActivationFunction = layer.ActivationFunction;
        }

        public double[] Forward(double[] input)
        {
            int neuronCount = Neurons.Length;
            double[] result = new double[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                result[i] = Neurons[i].Forward(ActivationFunction, input);
            }
            return result;
        }

        public double[] Forward(ILayerBase layer)
        {
            return Forward(layer.Output);
        }

        public void UpdateWeights(double alpha, double momentumRate)
        {
            foreach (var neuron in Neurons)
            {
                neuron.Update(alpha, momentumRate);
            }
        }
    }
}
