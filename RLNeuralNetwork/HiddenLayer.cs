using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLNeuralNetwork
{
    class HiddenLayer : NetworkLayer
    {
        public HiddenLayer(int inputs, int nodes, double baseLearningRate) : base(inputs, nodes, baseLearningRate) { }

        public HiddenLayer(double[,] weights, double baseLearningRate) : base(weights, baseLearningRate) { }

        internal override double activationFunction(double x)
        {
            //return x;
            double e2x = Math.Pow(Math.E, 2 * x);
            return (e2x - 1) / (e2x + 1);
        }

        internal override double activationDerivative(double x)
        {
            //return 1;
            double y = activationFunction(x);
            return 1 - (y * y);
        }
    }
}
