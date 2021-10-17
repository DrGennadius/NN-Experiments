namespace NNExperiments.Common
{
    /// <summary>
    /// Input and output range.
    /// </summary>
    public struct IORanges
    {
        public IORanges(Range<double> inputRange, Range<double> outputRange)
        {
            InputRange = inputRange;
            OutputRange = outputRange;
        }

        public Range<double> InputRange { get; set; }
        public Range<double> OutputRange { get; set; }
    }
}
