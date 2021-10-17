using System;

namespace NNExperiments.Common
{
    public class Interval<T> where T : struct, IComparable<T>
    {
        public Interval(T start, T end)
            : this()
        {
            Start = start;
            End = end;
        }

        public Interval()
        {
            IncludeStart = true;
            IncludeEnd = true;
        }

        public T? Start { get; set; }

        public T? End { get; set; }

        public bool IncludeStart { get; set; }

        public bool IncludeEnd { get; set; }

        public override string ToString()
        {
            return (IncludeStart ? "[" : "(")
                + (Start.HasValue ? Start.Value.ToString() : "Infinity")
                + ':'
                + (End.HasValue ? End.Value.ToString() : "Infinity")
                + (IncludeEnd ? "]" : ")");
        }

        public bool IsValid()
        {
            if (Start.HasValue && End.HasValue)
            {
                return Start.Value.CompareTo(End.Value) <= 0;
            }
            return true;
        }

        public bool ContainsValue(T value)
        {
            bool gStart = false, lEnd = false;
            if (Start.HasValue)
            {
                if (IncludeStart && Start.Value.CompareTo(value) <= 0 || Start.Value.CompareTo(value) < 0)
                {
                    gStart = true;
                }
            }
            else
            {
                gStart = true;
            }
            if (gStart)
            {
                if (End.HasValue)
                {
                    if (IncludeStart && value.CompareTo(End.Value) <= 0 || value.CompareTo(End.Value) < 0)
                    {
                        lEnd = true;
                    }
                }
                else
                {
                    lEnd = true;
                }
            }
            return gStart && lEnd;
        }

        public bool IsInsideInterval(Interval<T> interval)
        {
            return IsValid() && interval.IsValid() && interval.ContainsValue(Start.Value) && interval.ContainsValue(End.Value);
        }

        public bool ContainsInterval(Interval<T> interval)
        {
            return IsValid() && interval.IsValid() && ContainsValue(interval.Start.Value) && ContainsValue(interval.End.Value);
        }

        public Type GetValueType()
        {
            return typeof(T);
        }

        private string GetOperationForStart()
        {
            if (IncludeStart)
            {
                return "ge";
            }
            else
            {
                return "gt";
            }
        }

        private string GetOperationForEnd()
        {
            if (IncludeEnd)
            {
                return "le";
            }
            else
            {
                return "lt";
            }
        }
    }

    public class Interval : Interval<double>
    {
        public Interval(double start, double end)
            : base(start, end)
        {
        }

        public Interval()
            : base()
        {
        }

        public override string ToString()
        {
            return base.ToString();
        }
    }
}
