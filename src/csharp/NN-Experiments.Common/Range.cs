using System;

namespace NNExperiments.Common
{
    public struct Range<T> where T : struct, IComparable<T>
    {
        public Range(T min, T max)
        {
            Min = min;
            Max = max;
        }

        public T Min { get; set; }
        public T Max { get; set; }

        public override string ToString()
        {
            return string.Format("[{0},{1}]", Min, Max);
        }

        public Interval<T> GetInterval()
        {
            return new Interval<T>(Min, Max);
        }
    }
}
