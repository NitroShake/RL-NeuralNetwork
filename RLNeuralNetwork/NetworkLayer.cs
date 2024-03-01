using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLNeuralNetwork
{
    abstract class NetworkLayer
    {
        internal double[,] weights;
        internal double[] lastFeedForward;
        internal double[] losses;
        internal NetworkLayer previousLayer;
        internal double[] previousInputs;
        internal bool areLossesValid = false;
        internal double baseLearningRate;
        public NetworkLayer(int inputs, int nodes, double baseLearningRate)
        {
            Random r = new Random();
            weights = new double[inputs, nodes];
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    weights[i, j] = (double)r.NextDouble() * 0.000005;
                }
            }
            this.baseLearningRate = baseLearningRate;
        }
        public NetworkLayer(double[,] weights, double baseLearningRate)
        {
            previousLayer = null;
            this.weights = weights;
            this.baseLearningRate = baseLearningRate;
        }

        public virtual double[] feedforward(double[] previousLayer)
        {
            //TODO: throw exception for previousLayer/weight size mismatch
            double[] nodes = new double[weights.GetLength(1)];
            for (int i = 0; i < nodes.Length; i++)
            {
                for (int j = 0; j < weights.GetLength(0); j++)
                {
                    nodes[i] += previousLayer[j] * weights[j, i];
                    //.WriteLine($"adding {previousLayer[j]} * {weights[j, i]} to {nodes[i]}");
                }
                nodes[i] = activationFunction(nodes[i]);
                //Console.WriteLine(nodes[i]);
            }
            previousInputs = previousLayer;
            lastFeedForward = nodes;
            return nodes;
        }


        internal abstract double activationFunction(double x);

        internal abstract double activationDerivative(double x);
        internal virtual void backPropagate(double learningRateMulti = 1)
        {
            double learningRate = learningRateMulti * baseLearningRate;
            if (!areLossesValid)
            {
                throw new InvalidOperationException("Please begin backpropagation using backpropagate(losses) instead.");
            }
            areLossesValid = false;
            if (previousLayer != null)
            {
                previousLayer.losses = new double[weights.GetLength(0)];
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    for (int j = 0; j < weights.GetLength(1); j++)
                    {
                        previousLayer.losses[i] += weights[i, j] * losses[j];
                    }
                    previousLayer.losses[i] *= activationDerivative(previousLayer.lastFeedForward[i]);
                }
                previousLayer.areLossesValid = true;
            }
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    weights[i, j] = weights[i, j] - (learningRate * previousInputs[i] * losses[j]);
                }
            }
            if (previousLayer != null)
            {
                previousLayer.backPropagate(learningRateMulti: learningRateMulti);
            }
        }

        internal virtual void backPropagate(double[] losses, double learningRateMulti)
        {
            this.losses = losses;
            areLossesValid = true;
            backPropagate(learningRateMulti: learningRateMulti);
        }
    }
}
