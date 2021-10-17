using NNExperiments.Common.Training;
using NNExperiments.Perceptrons.Alternatives;
using NNExperiments.Perceptrons.Common;
using System;

namespace NNExperiments.Perceptrons
{
    public class PerceptronBase : IPerceptronBase
    {
        private double[][][] _weights;
        private double[][][] _prevWeightChanges;
        private double[][] _biases;
        private double[][] _biasChanges;
        private double[][] _outputs;
        private double[][] _derivatives;
        private double[][] _deltas;
        private double[] _input;
        private double _momentumRate;

        public double[][][] Weights { get => _weights; set => _weights = value; }

        public double[][] Biases { get => _biases; set => _biases = value; }

        public double MomentumRate { get => _momentumRate; set => _momentumRate = value; }
        public PerceptronTopology Topology { get; set; }

        public PerceptronBase()
        {
        }

        public PerceptronBase(PerceptronTopology topology)
        {
            Topology = topology;
            Initialize();
        }

        public PerceptronBase(PerceptronTopology topology, double momentumRate = 0.5)
        {
            Topology = topology;
            _momentumRate = momentumRate;
            Initialize();
        }

        public PerceptronBase(IPerceptronBase perceptron)
        {
            TransferFrom(perceptron);
        }

        public PerceptronBase(IPerceptronOld perceptronOld)
        {
            TransferFrom(perceptronOld);
        }

        public void Backward(double[] input, double[] targetOutput, double alpha)
        {
            int lastLayerIndex = Topology.GetLayerCount() - 1;

            // From end.
            for (int i = 0; i < targetOutput.Length; i++)
            {
                // Difference between goal output and real output elements.
                double e = _outputs[lastLayerIndex][i] - targetOutput[i];
                _deltas[lastLayerIndex][i] = e * _derivatives[lastLayerIndex][i];
            }

            // Calculate each previous delta based on the current one
            // by multiplying by the transposed matrix.
            for (int i = lastLayerIndex; i > 0; i--)
            {
                var prevLayerDeltas = _deltas[i - 1];
                var currentLayerDeltas = _deltas[i];
                var prevLayerWeights = _weights[i - 1];
                var currentLayerWeights = _weights[i];
                var prevLayerDerivatives = _derivatives[i - 1];
                int sizen = prevLayerWeights.Length;
                for (int n = 0; n < sizen; n++)
                {
                    prevLayerDeltas[n] = 0.0;
                }
                sizen = currentLayerWeights.Length;
                for (int n = 0; n < sizen; n++)
                {
                    var currentNeuronWeights = currentLayerWeights[n];
                    int sizew = currentNeuronWeights.Length;
                    for (int w = 0; w < sizew; w++)
                    {
                        prevLayerDeltas[w] += currentNeuronWeights[w] * currentLayerDeltas[n] * prevLayerDerivatives[w];
                    }
                }
            }

            // Update weights and biases.
            var layerWeights = _weights[0];
            var layerBiases = _biases[0];
            var layerDeltas = _deltas[0];
            var layerPrevWeightChanges = _prevWeightChanges[0];
            var layerBiasChanges = _biasChanges[0];
            int sizeN = layerWeights.Length;
            for (int n = 0; n < sizeN; n++)
            {
                var alphaDelta = -alpha * layerDeltas[n];
                var neuronWeights = layerWeights[n];
                var neuronPrevWeightChanges = layerPrevWeightChanges[n];
                int sizeW = neuronWeights.Length;
                for (int w = 0; w < sizeW; w++)
                {
                    double weightChange = alphaDelta * _input[w] + neuronPrevWeightChanges[w] * _momentumRate;
                    neuronPrevWeightChanges[w] = weightChange;
                    neuronWeights[w] += weightChange;
                }
                double biasChange = alphaDelta + layerBiasChanges[n] * _momentumRate;
                layerBiasChanges[n] = biasChange;
                layerBiases[n] += biasChange;
            }
            int sizeL = _weights.Length;
            for (int i = 1; i < sizeL; i++)
            {
                var prevOutputs = _outputs[i - 1];
                layerWeights = _weights[i];
                layerBiases = _biases[i];
                layerDeltas = _deltas[i];
                layerPrevWeightChanges = _prevWeightChanges[i];
                layerBiasChanges = _biasChanges[i];
                sizeN = layerWeights.Length;
                for (int n = 0; n < sizeN; n++)
                {
                    var alphaDelta = -alpha * layerDeltas[n];
                    var neuronWeights = layerWeights[n];
                    var neuronPrevWeightChanges = layerPrevWeightChanges[n];
                    int sizeW = neuronWeights.Length;
                    for (int w = 0; w < sizeW; w++)
                    {
                        double weightChange = alphaDelta * prevOutputs[w] + neuronPrevWeightChanges[w] * _momentumRate;
                        neuronPrevWeightChanges[w] = weightChange;
                        neuronWeights[w] += weightChange;
                    }
                    double biasChange = alphaDelta + layerBiasChanges[n] * _momentumRate;
                    layerBiasChanges[n] = biasChange;
                    layerBiases[n] += biasChange;
                }
            }
        }

        public object Clone()
        {
            PerceptronBase perceptronBase = new();
            perceptronBase.TransferFrom(this);
            return perceptronBase;
        }

        public double[] Forward(double[] input)
        {
            _input = input;
            int sizeL = _weights.Length;
            for (int i = 0; i < sizeL; i++)
            {
                var activationFunction = Topology.ActivationFunctions[i].Calculate;
                var derivativeActivationFunction = Topology.ActivationFunctions[i].CalculateDerivative;
                var layerOutputs = _outputs[i];
                var layerDerivatives = _derivatives[i];
                var layerBiases = _biases[i];
                var layerWeights = _weights[i];
                int sizeN = layerWeights.Length;
                for (int n = 0; n < sizeN; n++)
                {
                    double sum = layerBiases[n];
                    var neuronWeights = layerWeights[n];
                    int sizeW = neuronWeights.Length;
                    for (int w = 0; w < sizeW; w++)
                    {
                        sum += neuronWeights[w] * input[w];
                    }
                    layerOutputs[n] = activationFunction(sum);
                    layerDerivatives[n] = derivativeActivationFunction(layerOutputs[n]);
                }
                input = layerOutputs;
            }
            return input;
        }

        public TrainStats Train(double[,] inputs, double[,] outputs, double alpha, double targetError, int maxEpoch, bool printError = false)
        {
            return Train(new TrainData(inputs, outputs), alpha, targetError, maxEpoch, printError);
        }

        public TrainStats Train(TrainData trainData, double alpha, double targetError, int maxEpoch, bool printError = false)
        {

            throw new NotImplementedException();
        }

        public void TransferFrom(IPerceptronBase otherPerceptron)
        {
            _weights = new double[otherPerceptron.Weights.Length][][];
            for (int i = 0; i < otherPerceptron.Weights.Length; i++)
            {
                _weights[i] = new double[otherPerceptron.Weights[i].Length][];
                for (int n = 0; n < otherPerceptron.Weights[i].Length; n++)
                {
                    _weights[i][n] = new double[otherPerceptron.Weights[i][n].Length];
                    for (int w = 0; w < otherPerceptron.Weights[i][n].Length; w++)
                    {
                        _weights[i][n][w] = otherPerceptron.Weights[i][n][w];
                    }
                }
            }
            _biases = new double[otherPerceptron.Biases.Length][];
            for (int i = 0; i < otherPerceptron.Biases.Length; i++)
            {
                _biases[i] = new double[otherPerceptron.Biases[i].Length];
                for (int n = 0; n < otherPerceptron.Biases[i].Length; n++)
                {
                    _biases[i][n] = otherPerceptron.Biases[i][n];
                }
            }
            Topology = otherPerceptron.Topology;
            _momentumRate = otherPerceptron.MomentumRate;
            int[] sizes = otherPerceptron.Topology.GetSizes();
            InitializeInternalValues(sizes);
        }

        public void TransferFrom(IPerceptronOld perceptronOld)
        {
            double[][][] otherWeights = perceptronOld.GetWeights();
            int[] sizes = new int[otherWeights.Length + 1];
            sizes[0] = otherWeights[0][0].Length;
            for (int i = 0; i < otherWeights.Length; i++)
            {
                sizes[i + 1] = otherWeights[i].Length;
            }
            Topology = new PerceptronTopology(sizes, perceptronOld.GetActivationFunction());
            _weights = new double[otherWeights.Length][][];
            double[][] otherBiases = perceptronOld.GetBiases();
            bool haveBiases = otherBiases.Length > 0;
            _biases = haveBiases ? perceptronOld.GetBiases() : new double[otherWeights.Length][];
            for (int i = 1; i < sizes.Length; i++)
            {
                _weights[i - 1] = new double[sizes[i]][];
                for (int n = 0; n < sizes[i]; n++)
                {
                    _weights[i - 1][n] = new double[sizes[i - 1]];
                    for (int w = 0; w < sizes[i - 1]; w++)
                    {
                        _weights[i - 1][n][w] = otherWeights[i - 1][n][w];
                    }
                }
                if (!haveBiases)
                {
                    _biases[i - 1] = new double[sizes[i]];
                }
            }
            _momentumRate = perceptronOld.GetMomentumRate();
            InitializeInternalValues(sizes);
        }

        public void TransferTo(IPerceptronBase otherPerceptron)
        {
            otherPerceptron.TransferFrom(this);
        }

        private void Initialize()
        {
            Random random = new(DateTime.Now.Millisecond);

            int[] sizes = Topology.GetSizes();
            int layerCount = sizes.Length - 1;

            _weights = new double[layerCount][][];
            _biases = new double[layerCount][];

            for (int i = 1; i < sizes.Length; i++)
            {
                _weights[i - 1] = new double[sizes[i]][];
                _biases[i - 1] = new double[sizes[i]];
                for (int n = 0; n < sizes[i]; n++)
                {
                    _weights[i - 1][n] = new double[sizes[i - 1]];
                    for (int w = 0; w < sizes[i - 1]; w++)
                    {
                        _weights[i - 1][n][w] = random.NextDouble() - 0.5;
                    }
                    _biases[i - 1][n] = random.NextDouble() - 0.5;
                }
            }

            InitializeInternalValues(sizes);
        }

        private void InitializeInternalValues(int[] sizes)
        {
            int layerCount = sizes.Length - 1;

            _prevWeightChanges = new double[layerCount][][];
            _biasChanges = new double[layerCount][];
            _outputs = new double[layerCount][];
            _derivatives = new double[layerCount][];
            _deltas = new double[layerCount][];

            for (int i = 1; i < sizes.Length; i++)
            {
                _prevWeightChanges[i - 1] = new double[sizes[i]][];
                _biasChanges[i - 1] = new double[sizes[i]];
                for (int n = 0; n < sizes[i]; n++)
                {
                    _prevWeightChanges[i - 1][n] = new double[sizes[i - 1]];
                    for (int w = 0; w < sizes[i - 1]; w++)
                    {
                        _prevWeightChanges[i - 1][n][w] = 0;
                    }
                    _biasChanges[i - 1][n] = 0;
                }
                _outputs[i - 1] = new double[sizes[i]];
                _derivatives[i - 1] = new double[sizes[i]];
                _deltas[i - 1] = new double[sizes[i]];
            }
        }
    }
}
