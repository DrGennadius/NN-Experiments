/****************************************************************************************
 * Based on https://programforyou.ru/poleznoe/pishem-neuroset-pryamogo-rasprostraneniya *
 ****************************************************************************************/

using NNExperiments.Common.Training;
using System;

namespace NNExperiments.Perceptrons.Alternatives.V1
{
    /// <summary>
    /// Perceptron. Class based on programforyou.ru sample
    /// </summary>
    public class PerceptronAV1
    {
        private struct LayerT
        {
            public VectorAV1 Input; // layer input
            public VectorAV1 Output; // activated layer output
            public VectorAV1 DerivatedOutput; // derivative of layer activation function
        }

        private MatrixAV1[] _weights; // layer weight matrices
        private LayerT[] _layerValues; // values on each layer
        private VectorAV1[] _deltas; // error deltas on each layer

        private int _layerCount; // number of layers

        public PerceptronAV1(int[] sizes)
        {
            Random random = new(DateTime.Now.Millisecond);

            _layerCount = sizes.Length - 1;

            _weights = new MatrixAV1[_layerCount];
            _layerValues = new LayerT[_layerCount];
            _deltas = new VectorAV1[_layerCount];

            for (int k = 1; k < sizes.Length; k++)
            {
                _weights[k - 1] = new MatrixAV1(sizes[k], sizes[k - 1], random);

                _layerValues[k - 1].Input = new VectorAV1(sizes[k - 1]);
                _layerValues[k - 1].Output = new VectorAV1(sizes[k]);
                _layerValues[k - 1].DerivatedOutput = new VectorAV1(sizes[k]);

                _deltas[k - 1] = new VectorAV1(sizes[k]);
            }
        }

        public PerceptronAV1(IPerceptronOld perceptron)
        {
            double[][][] otherWeights = perceptron.GetWeights();
            _layerCount = otherWeights.Length;

            _weights = new MatrixAV1[_layerCount];
            _layerValues = new LayerT[_layerCount];
            _deltas = new VectorAV1[_layerCount];

            int[] sizes = new int[_layerCount + 1];
            sizes[0] = otherWeights[0][0].Length;
            for (int i = 1; i <= _layerCount; i++)
            {
                sizes[i] = otherWeights[i - 1].Length;
            }

            for (int i = 1; i < sizes.Length; i++)
            {
                _weights[i - 1] = new MatrixAV1(otherWeights[i - 1]);

                _layerValues[i - 1].Input = new VectorAV1(sizes[i - 1]);
                _layerValues[i - 1].Output = new VectorAV1(sizes[i]);
                _layerValues[i - 1].DerivatedOutput = new VectorAV1(sizes[i]);

                _deltas[i - 1] = new VectorAV1(sizes[i]);
            }
        }

        public VectorAV1 Forward(VectorAV1 input)
        {
            for (int k = 0; k < _layerCount; k++)
            {
                if (k == 0)
                {
                    for (int i = 0; i < input.Length; i++)
                    {
                        _layerValues[k].Input[i] = input[i];
                    }
                }
                else
                {
                    for (int i = 0; i < _layerValues[k - 1].Output.Length; i++)
                    {
                        _layerValues[k].Input[i] = _layerValues[k - 1].Output[i];
                    }
                }

                for (int i = 0; i < _weights[k].RowSize; i++)
                {
                    double y = 0;

                    for (int j = 0; j < _weights[k].ColumnSize; j++)
                    {
                        y += _weights[k][i, j] * _layerValues[k].Input[j];
                    }

                    // activation by sigmoid function
                    _layerValues[k].Output[i] = 1 / (1 + Math.Exp(-y));
                    _layerValues[k].DerivatedOutput[i] = _layerValues[k].Output[i] * (1 - _layerValues[k].Output[i]);

                    // activation by hyperbolic tangent function
                    //LayerValues[k].Output[i] = Math.Tanh(y);
                    //LayerValues[k].DerivatedOutput[i] = 1 - L[k].z[i] * L[k].z[i];

                    // activation by ReLU function
                    //LayerValues[k].Output[i] = y > 0 ? y : 0;
                    //LayerValues[k].DerivatedOutput[i] = y > 0 ? 1 : 0;
                }
            }

            return _layerValues[_layerCount - 1].Output;
        }

        public void Backward(VectorAV1 output, ref double error)
        {
            int last = _layerCount - 1;

            error = 0;

            for (int i = 0; i < output.Length; i++)
            {
                double e = _layerValues[last].Output[i] - output[i];

                _deltas[last][i] = e * _layerValues[last].DerivatedOutput[i];
                error += e * e / 2;
            }

            // Calculate each previous delta based on the current one
            // by multiplying by the transposed matrix.
            for (int k = last; k > 0; k--)
            {
                for (int i = 0; i < _weights[k].ColumnSize; i++)
                {
                    _deltas[k - 1][i] = 0;

                    for (int j = 0; j < _weights[k].RowSize; j++)
                    {
                        _deltas[k - 1][i] += _weights[k][j, i] * _deltas[k][j];
                    }

                    // multiply the resulting value by the derivative of the previous layer
                    _deltas[k - 1][i] *= _layerValues[k - 1].DerivatedOutput[i];
                }
            }
        }

        /// <summary>
        /// Update weights
        /// </summary>
        /// <param name="alpha">Learning rate</param>
        public void UpdateWeights(double alpha)
        {
            for (int k = 0; k < _layerCount; k++)
            {
                for (int i = 0; i < _weights[k].RowSize; i++)
                {
                    for (int j = 0; j < _weights[k].ColumnSize; j++)
                    {
                        _weights[k][i, j] -= alpha * _deltas[k][i] * _layerValues[k].Input[j];
                    }
                }
            }
        }

        /// <summary>
        /// Train
        /// </summary>
        /// <param name="X">Inputs</param>
        /// <param name="Y">Outputs</param>
        /// <param name="alpha">Learning rate</param>
        /// <param name="eps">Target error</param>
        /// <param name="epochs">Epoch number limit</param>
        public TrainStats Train(VectorAV1[] X, VectorAV1[] Y, double alpha, double eps, int epochs)
        {
            int epoch = 1; // epoch number

            double error; // epoch error

            do
            {
                error = 0;

                // Go through all the elements of the training set
                for (int i = 0; i < X.Length; i++)
                {
                    Forward(X[i]);
                    Backward(Y[i], ref error);
                    UpdateWeights(alpha);
                }

                // Console.WriteLine("epoch: {0}, error: {1}", epoch, error);

                epoch++;
            } while (epoch < epochs && error > eps);
            return new TrainStats
            {
                LastError = error,
                NumberOfEpoch = epoch
            };
        }
    }
}
