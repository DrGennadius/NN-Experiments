using NNExperiments.Common.ActivationFunctions;
using NNExperiments.Common.Training;
using NNExperiments.Perceptrons.Common;
using System;
using System.Linq;

namespace NNExperiments.Perceptrons.Alternatives
{
    /// <summary>
    /// Simple perceptron (without bias and momentum).
    /// </summary>
    public class SimplePerceptron : IPerceptronOld
    {
        readonly SimpleLayer[] Layers;

        public double[][][] Weights { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public double[][] Biases { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public PerceptronTopology Topology
        {
            get
            {
                return new PerceptronTopology(Layers[0].Neurons[0].Weights.Length, Layers.Select(x => x.Neurons.Length).ToArray(), GetActivationFunction());
            }
            set => throw new NotImplementedException();
        }

        public double MomentumRate { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        /// <summary>
        /// Create simple perceptron (without bias and momentum) by configuration.
        /// </summary>
        /// <param name="neuronsPerLayer">Configuration.</param>
        public SimplePerceptron(params int[] neuronsPerLayer)
            : this(neuronsPerLayer, new Random())
        {
        }

        /// <summary>
        /// Create simple perceptron (without bias and momentum) by configuration and set Random.
        /// </summary>
        /// <param name="neuronsPerLayer">Configuration.</param>
        /// <param name="random"></param>
        public SimplePerceptron(int[] neuronsPerLayer, Random random)
        {
            Layers = new SimpleLayer[neuronsPerLayer.Length - 1];

            for (int i = 1; i < neuronsPerLayer.Length; i++)
            {
                Layers[i - 1] = new SimpleLayer(neuronsPerLayer[i], neuronsPerLayer[i - 1], random);
            }
        }

        /// <summary>
        /// Create simple perceptron (without bias and momentum) by other Perceptron.
        /// </summary>
        /// <param name="perceptron">Other Perceptron.</param>
        public SimplePerceptron(IPerceptronOld perceptron)
        {
            double[][][] otherWeights = perceptron.GetWeights();
            int layerCount = otherWeights.GetLength(0);

            Layers = new SimpleLayer[layerCount];

            for (int i = 0; i < layerCount; i++)
            {
                Layers[i] = new SimpleLayer(otherWeights[i]);
            }
        }

        public double[] Forward(double[] input)
        {
            double[] result = input;
            for (int i = 0; i < Layers.Length; i++)
            {
                result = Layers[i].FeedForward(result);
            }
            return result;
        }

        public void Backward(double[] input, double[] targetOutput, double learningRate)
        {
            double[][] deltas = new double[Layers.Length][];
            int lastLayerIndex = Layers.Length - 1;

            // From end.
            deltas[lastLayerIndex] = new double[Layers[lastLayerIndex].Neurons.Length];
            for (int i = 0; i < targetOutput.Length; i++)
            {
                // Difference between goal output and real output elements.
                double e = Layers[lastLayerIndex].Neurons[i].Output - targetOutput[i];
                deltas[lastLayerIndex][i] = e * Layers[lastLayerIndex].Neurons[i].DerivatedOutput;
            }

            // Calculate each previous delta based on the current one
            // by multiplying by the transposed matrix.
            for (int k = lastLayerIndex; k > 0; k--)
            {
                var layer = Layers[k].Neurons;
                var previousLayer = Layers[k - 1].Neurons;

                deltas[k - 1] = new double[previousLayer.Length];

                for (int i = 0; i < previousLayer.Length; i++)
                {
                    deltas[k - 1][i] = 0.0;
                }
                for (int i = 0; i < layer.Length; i++)
                {
                    for (int j = 0; j < layer[i].Weights.Length; j++)
                    {
                        double w = layer[i].Weights[j];
                        double d = deltas[k][i];
                        deltas[k - 1][j] += w * d * previousLayer[j].DerivatedOutput;
                    }
                }

                //for (int i = 0; i < layer[0].Weights.Length; i++)
                //{
                //    deltas[k - 1][i] = 0;
                //    for (int j = 0; j < layer.Length; j++)
                //    {
                //        deltas[k - 1][i] += Layers[k].Neurons[j].Weights[i] * deltas[k][j];
                //    }
                //    deltas[k - 1][i] *= Layers[k - 1].Neurons[i].DerivatedOutput;
                //}
            }

            // Correcting weights.
            for (int i = 0; i < lastLayerIndex + 1; i++)
            {
                var layer = Layers[i].Neurons;
                for (int n = 0; n < layer.Length; n++)
                {
                    var neuron = layer[n];
                    for (int w = 0; w < neuron.Weights.Length; w++)
                    {
                        neuron.Weights[w] -= learningRate * deltas[i][n] * neuron.Input[w];
                    }
                }
            }
        }

        /// <summary>
        /// Training Perceptron.
        /// </summary>
        /// <param name="trainData">Data for training.</param>
        /// <param name="alpha">Learning rate.</param>
        /// <param name="targetError">Target error value.</param>
        /// <param name="maxEpoch">Max number epochs.</param>
        /// <param name="printError">Print error each epoch.</param>
        /// <returns></returns>
        public TrainStats Train(TrainData trainData, double alpha, double targetError, int maxEpoch, bool printError = false)
        {
            PerceptronTrainer perceptronTrainer = new();
            return perceptronTrainer.Train(this, trainData, alpha, targetError, maxEpoch, printError);
        }

        public void TransferWeightsFrom(IPerceptronOld otherPerceptron)
        {
            SetWeights(otherPerceptron.GetWeights());
        }

        public double[][][] GetWeights()
        {
            double[][][] weights = new double[Layers.Length][][];
            for (int i = 0; i < Layers.Length; i++)
            {
                weights[i] = new double[Layers[i].Neurons.Length][];
                for (int n = 0; n < Layers[i].Neurons.Length; n++)
                {
                    weights[i][n] = new double[Layers[i].Neurons[n].Weights.Length];
                    for (int w = 0; w < Layers[i].Neurons[n].Weights.Length; w++)
                    {
                        weights[i][n][w] = Layers[i].Neurons[n].Weights[w];
                    }
                }
            }
            return weights;
        }

        public void SetWeights(double[][][] weights)
        {
            int otherLayerSize = weights.Length;
            if (otherLayerSize != Layers.Length)
            {
                throw new ArgumentException("Sizes of perceptrons are not equal");
            }
            for (int i = 0; i < otherLayerSize; i++)
            {
                int otherNeuronSize = weights[i].GetLength(0);
                if (otherNeuronSize != Layers[i].Neurons.Length)
                {
                    throw new ArgumentException("Sizes of perceptrons are not equal");
                }
                for (int n = 0; n < otherNeuronSize; n++)
                {
                    int otherWeightsSize = weights[i][n].GetLength(0);
                    if (otherWeightsSize != Layers[i].Neurons[n].Weights.Length)
                    {
                        throw new ArgumentException("Sizes of perceptrons are not equal");
                    }
                }
            }
            for (int i = 0; i < Layers.Length; i++)
            {
                for (int n = 0; n < Layers[i].Neurons.Length; n++)
                {
                    for (int w = 0; w < Layers[i].Neurons[n].Weights.Length; w++)
                    {
                        Layers[i].Neurons[n].Weights[w] = weights[i][n][w];
                    }
                }
            }
        }

        public double[][] GetBiases()
        {
            return Array.Empty<double[]>();
        }

        public ActivationFunction GetActivationFunction()
        {
            return new ActivationFunction(ActivationFunctionType.Sigmoid);
        }

        public double GetMomentumRate()
        {
            return 0;
        }

        public TrainStats Train(double[,] inputs, double[,] outputs, double alpha, double targetError, int maxEpoch, bool printError = false)
        {
            throw new NotImplementedException();
        }

        public void TransferFrom(IPerceptronBase otherPerceptron)
        {
            throw new NotImplementedException();
        }

        public void TransferTo(IPerceptronBase otherPerceptron)
        {
            throw new NotImplementedException();
        }

        public object Clone()
        {
            throw new NotImplementedException();
        }
    }
}
