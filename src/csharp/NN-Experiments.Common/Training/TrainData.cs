using System;

namespace NNExperiments.Common.Training
{
    public class TrainData<T>
    {
        public TrainData()
        {
        }

        public TrainData(T[,] inputs, T[,] outputs)
        {
            Set(inputs, outputs);
        }

        public T[,] Inputs { get; private set; }

        public T[,] Outputs { get; private set; }

        /// <summary>
        /// Number of train items.
        /// </summary>
        public int Count
        {
            get => Inputs.GetLength(0);
        }

        public void Set(T[,] inputs, T[,] outputs)
        {
            if (inputs.GetLength(0) != outputs.GetLength(0))
            {
                throw new ArgumentException();
            }
            Inputs = inputs;
            Outputs = outputs;
        }

        public T[] GetInputRow(int index)
        {
            int inputLength = Inputs.GetLength(1);
            T[] row = new T[inputLength];
            for (int i = 0; i < inputLength; i++)
            {
                row[i] = Inputs[index, i];
            }
            return row;
        }

        public T[] GetOutputRow(int index)
        {
            int outputLength = Outputs.GetLength(1);
            T[] row = new T[outputLength];
            for (int i = 0; i < outputLength; i++)
            {
                row[i] = Outputs[index, i];
            }
            return row;
        }

        public T[][] GetRow(int index)
        {
            return new T[][] { GetInputRow(index), GetOutputRow(index) };
        }
    }

    /// <summary>
    /// Data for training.
    /// </summary>
    public class TrainData : TrainData<double>
    {
        public TrainData()
        {
        }

        public TrainData(double[,] inputs, double[,] outputs)
            : base(inputs, outputs)
        {
        }

        public TrainData GetNormalized(Range<double> inputRange, Range<double> outputRange)
        {
            return CommonFunctions.CreateNewNormalizeTrainData(this, inputRange, outputRange);
        }

        /// <summary>
        /// Generate data "XOR".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataXOR()
        {
            return new TrainData(
                new double[,]
                {
                    { 0, 0 },
                    { 0, 1 },
                    { 1, 0 },
                    { 1, 1 }
                },
                new double[,]
                {
                    { 0 },
                    { 1 },
                    { 1 },
                    { 0 }
                }
            );
        }

        /// <summary>
        /// Generate data "OR".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataOR()
        {
            return new TrainData(
                new double[,]
                {
                    { 0, 0 },
                    { 0, 1 },
                    { 1, 0 },
                    { 1, 1 }
                },
                new double[,]
                {
                    { 0 },
                    { 1 },
                    { 1 },
                    { 1 }
                }
            );
        }

        /// <summary>
        /// Generate data "AND".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataAND()
        {
            return new TrainData(
                new double[,]
                {
                    { 0, 0 },
                    { 0, 1 },
                    { 1, 0 },
                    { 1, 1 }
                },
                new double[,]
                {
                    { 0 },
                    { 0 },
                    { 0 },
                    { 1 }
                }
            );
        }

        /// <summary>
        /// Generate data "NAND".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataNAND()
        {
            return new TrainData(
                new double[,]
                {
                    { 0, 0 },
                    { 0, 1 },
                    { 1, 0 },
                    { 1, 1 }
                },
                new double[,]
                {
                    { 1 },
                    { 1 },
                    { 1 },
                    { 0 }
                }
            );
        }

        /// <summary>
        /// Generate data "Multiplication".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataMultiplication()
        {
            int min = 0;
            int max = 9;
            int p = 0;
            for (int i = min; i <= max; i++)
            {
                p++;
            }
            p *= p;
            double[,] inputs = new double[p, 2];
            double[,] outputs = new double[p, 1];
            int count = 0;
            for (int i = min; i <= max; i++)
            {
                for (int k = min; k <= max; k++)
                {
                    inputs[count, 0] = i;
                    inputs[count, 1] = k;
                    outputs[count, 0] = i * k;
                    count++;
                }
            }
            return new TrainData(inputs, outputs);
        }

        /// <summary>
        /// Generate data "Multiplication".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataMultiplication(double[,] inputs)
        {
            double[,] outputs = new double[inputs.GetLength(0), 1];
            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                outputs[i, 0] = 1;
                for (int k = 0; k < inputs.GetLength(1); k++)
                {
                    outputs[i, 0] *= inputs[i, k];
                }
            }
            return new TrainData(inputs, outputs);
        }

        /// <summary>
        /// Generate data "Simple Numbers".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataSimpleNumbers()
        {
            return new TrainData(
                new double[,]
                {
                    {
                        0, 0, 0, 0, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 0, 0
                    },
                    {
                        0, 0, 0, 0, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 1, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0
                    },
                    {
                        0, 0, 0, 0, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 0, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 0, 0
                    },
                    {
                        0, 0, 0, 0, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 0, 0
                    },
                    {
                        0, 0, 0, 0, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0
                    },
                    {
                        0, 0, 0, 0, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 0, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 0, 0
                    },
                    {
                        0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 0, 0
                    },
                    {
                        0, 0, 0, 0, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0
                    },
                    {
                        0, 0, 0, 0, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 0, 0
                    },
                    {
                        0, 0, 0, 0, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0
                    }
                },
                new double[,]
                {
                    { 0 },
                    { 1 },
                    { 2 },
                    { 3 },
                    { 4 },
                    { 5 },
                    { 6 },
                    { 7 },
                    { 8 },
                    { 9 }
                }
            );
        }
    }
}
