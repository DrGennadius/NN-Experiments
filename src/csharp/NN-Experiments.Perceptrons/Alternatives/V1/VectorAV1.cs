/****************************************************************************************
 * Based on https://programforyou.ru/poleznoe/pishem-neuroset-pryamogo-rasprostraneniya *
 ****************************************************************************************/

namespace NNExperiments.Perceptrons.Alternatives.V1
{
    /// <summary>
    /// Vector. Class based on programforyou.ru sample
    /// </summary>
    public class VectorAV1
    {
        private double[] _values;

        public int Length;

        /// <summary>
        /// Constructor with length.
        /// </summary>
        /// <param name="length"></param>
        public VectorAV1(int length)
        {
            Length = length;
            _values = new double[length];
        }

        /// <summary>
        /// Constructor with array of value.
        /// </summary>
        /// <param name="values"></param>
        public VectorAV1(params double[] values)
        {
            Length = values.Length;
            _values = new double[Length];

            for (int i = 0; i < Length; i++)
            {
                _values[i] = values[i];
            }
        }

        public double this[int index]
        {
            get { return _values[index]; }
            set { _values[index] = value; }
        }
    }
}
