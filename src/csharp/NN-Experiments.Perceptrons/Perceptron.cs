using NNExperiments.Common.ActivationFunctions;
using NNExperiments.Common.Training;
using NNExperiments.Perceptrons.Common;
using NNExperiments.Perceptrons.Layers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NNExperiments.Perceptrons
{
    public class Perceptron : IPerceptron
    {
        public IList<ILayerBase> Layers { get; set; }

        public double[][][] Weights
        {
            get
            {
                return Layers.Select(x => x.Weights).ToArray();
            }
            set
            {
                throw new NotImplementedException();
            }
        }

        public double[][] Biases
        {
            get
            {
                return Layers.Select(x => x.Neurons.Select(n => n.Bias).ToArray()).ToArray();
            }
            set
            {
                throw new NotImplementedException();
            }
        }

        public PerceptronTopology Topology { get; set; }

        public double MomentumRate { get; set; }

        public Perceptron()
        {
            Layers = new List<ILayerBase>();
        }

        public Perceptron(PerceptronTopology topology)
        {
            Topology = topology;
            Layers = new List<ILayerBase>(Topology.GetLayerCount());
            Initialize();
        }

        /// <summary>
        /// Create perceptron with bias and momentum by other Perceptron.
        /// </summary>
        /// <param name="perceptron"></param>
        /// <param name="momentumRate"></param>
        public Perceptron(IPerceptronBase perceptron)
        {
            Layers = new List<ILayerBase>(perceptron.Topology.GetLayerCount());
            TransferFrom(perceptron);
        }

        /// <summary>
        /// Create perceptron with bias and momentum by other Perceptron and setting momentum rate.
        /// </summary>
        /// <param name="perceptron"></param>
        /// <param name="momentumRate"></param>
        public Perceptron(IPerceptronBase perceptron, double momentumRate = 0.5)
        {
            TransferFrom(perceptron);
            MomentumRate = momentumRate;
        }

        public IPerceptron Add(ILayerBase layer)
        {
            Layers.Add(layer);
            return this;
        }

        public IPerceptron Add(IEnumerable<ILayerBase> layers)
        {
            foreach (var item in layers)
            {
                Layers.Add(item);
            }
            return this;
        }

        public IPerceptron AddLayer(double[][] layerWeights, double[] layerBias)
        {
            return Add(new Layer(layerWeights, layerBias));
        }

        public IPerceptron AddLayer(double[][] layerWeights, double[] layerBias, ActivationFunction activationFunction)
        {
            return Add(new Layer(layerWeights, layerBias, activationFunction));
        }

        public IPerceptron AddLayer(double[][] layerWeights, Random random)
        {
            return Add(new Layer(layerWeights, random));
        }

        public IPerceptron AddLayer(double[][] layerWeights, ActivationFunction activationFunction, Random random)
        {
            return Add(new Layer(layerWeights, activationFunction, random));
        }

        public IPerceptron AddLayer(double[,] layerWeights, double[] layerBias)
        {
            return Add(new Layer(layerWeights, layerBias));
        }

        public IPerceptron AddLayer(double[,] layerWeights, double[] layerBias, ActivationFunction activationFunction)
        {
            return Add(new Layer(layerWeights, layerBias, activationFunction));
        }

        public IPerceptron AddLayer(double[,] layerWeights, Random random)
        {
            return Add(new Layer(layerWeights, random));
        }

        public IPerceptron AddLayer(double[,] layerWeights, ActivationFunction activationFunction, Random random)
        {
            return Add(new Layer(layerWeights, activationFunction, random));
        }

        public IPerceptron AddLayer(int numberOfNeurons, int numberOfInputs, Random random)
        {
            return Add(new Layer(numberOfNeurons, numberOfInputs, random));
        }

        public IPerceptron AddLayer(int numberOfNeurons, int numberOfInputs, ActivationFunction activationFunction, Random random)
        {
            return Add(new Layer(numberOfNeurons, numberOfInputs, activationFunction, random));
        }

        public void Backward(double[] input, double[] targetOutput, double alpha)
        {
            int lastLayerIndex = Layers.Count - 1;

            // From end.
            for (int i = 0; i < targetOutput.Length; i++)
            {
                // Difference between goal output and real output elements.
                double e = Layers[lastLayerIndex].Neurons[i].Output - targetOutput[i];
                Layers[lastLayerIndex].Neurons[i].Delta = e * Layers[lastLayerIndex].Neurons[i].DerivatedOutput;
            }

            // Calculate each previous delta based on the current one
            // by multiplying by the transposed matrix.
            for (int k = lastLayerIndex; k > 0; k--)
            {
                var layer = Layers[k].Neurons;
                var previousLayer = Layers[k - 1].Neurons;
                for (int i = 0; i < previousLayer.Length; i++)
                {
                    previousLayer[i].Delta = 0.0;
                }
                for (int i = 0; i < layer.Length; i++)
                {
                    var neuron = layer[i];
                    for (int j = 0; j < neuron.Weights.Length; j++)
                    {
                        previousLayer[j].Delta += neuron.Weights[j] * neuron.Delta * previousLayer[j].DerivatedOutput;
                    }
                }
            }

            // Correcting weights and bias.
            UpdateWeights(alpha);
        }

        public IPerceptron Build()
        {
            throw new NotImplementedException();
        }

        public object Clone()
        {
            return new Perceptron(this);
        }

        public double[] Forward(double[] input)
        {
            int layerCount = Layers.Count;
            double[] output = Layers[0].Forward(input);
            for (int i = 1; i < layerCount; i++)
            {
                output = Layers[i].Forward(output);
            }
            return output;
        }

        public TrainStats Train(double[,] inputs, double[,] outputs, double alpha, double targetError, int maxEpoch, bool printError = false)
        {
            return Train(new TrainData(inputs, outputs), alpha, targetError, maxEpoch, printError);
        }

        public TrainStats Train(TrainData trainData, double alpha, double targetError, int maxEpoch, bool printError = false)
        {
            PerceptronTrainer trainer = new();
            return trainer.Train(this, trainData, alpha, targetError, maxEpoch, printError);
        }

        public void TransferFrom(IPerceptronBase otherPerceptron)
        {
            Topology = otherPerceptron.Topology;
            MomentumRate = otherPerceptron.MomentumRate;
            int layerCount = Topology.GetLayerCount();
            if (otherPerceptron.Biases == null)
            {
                Random random = new(DateTime.Now.Millisecond);
                for (int i = 0; i < layerCount; i++)
                {
                    Layers.Add(new Layer(otherPerceptron.Weights[i], Topology.ActivationFunctions[i], random));
                }
            }
            else
            {
                for (int i = 0; i < layerCount; i++)
                {
                    Layers.Add(new Layer(otherPerceptron.Weights[i], otherPerceptron.Biases[i], Topology.ActivationFunctions[i]));
                }
            }
        }

        public void TransferTo(IPerceptronBase otherPerceptron)
        {
            otherPerceptron.TransferFrom(this);
        }

        private void Initialize()
        {
            Random random = new(DateTime.Now.Millisecond);
            int[] sizes = Topology.GetSizes();
            for (int i = 1; i < sizes.Length; i++)
            {
                Layers.Add(new Layer(sizes[i], sizes[i - 1], random));
            }
        }

        private void UpdateWeights(double alpha)
        {
            foreach (var layer in Layers)
            {
                layer.UpdateWeights(alpha, MomentumRate);
            }
        }
    }
}
