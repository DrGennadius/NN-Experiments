/****************************************************************************************
 * Based on https://programforyou.ru/poleznoe/pishem-neuroset-pryamogo-rasprostraneniya *
 ****************************************************************************************/

using System;

namespace NNExperiments.Perceptrons.Alternatives.V1
{
    /// <summary>
    /// Matrix. Class based on programforyou.ru sample
    /// </summary>
    public class MatrixAV1
    {
        private double[][] _values;

        public int RowSize;
        public int ColumnSize;

        /// <summary>
        /// Creation of a matrix of a given size and filling with random numbers from the interval (-0.5, 0.5).
        /// </summary>
        /// <param name="rowSize"></param>
        /// <param name="columnSize"></param>
        /// <param name="random"></param>
        public MatrixAV1(int rowSize, int columnSize, Random random)
        {
            RowSize = rowSize;
            ColumnSize = columnSize;

            _values = new double[rowSize][];

            for (int i = 0; i < rowSize; i++)
            {
                _values[i] = new double[columnSize];

                for (int j = 0; j < columnSize; j++)
                {
                    _values[i][j] = random.NextDouble() - 0.5;
                }
            }
        }

        /// <summary>
        /// Creation of a matrix by data.
        /// </summary>
        /// <param name="data"></param>
        public MatrixAV1(double[][] data)
        {
            _values = data;

            RowSize = data.Length;
            ColumnSize = data[0].Length;
        }

        public double this[int i, int j]
        {
            get { return _values[i][j]; }
            set { _values[i][j] = value; }
        }
    }
}
