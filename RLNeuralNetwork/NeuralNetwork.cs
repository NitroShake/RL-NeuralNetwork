using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLNeuralNetwork
{
    class NeuralNetwork
    {
        internal HiddenLayer[] hiddenLayers;
        internal OutputLayer outputLayer;

        public NeuralNetwork(HiddenLayer[] hiddenLayers, OutputLayer outputLayer)
        {
            this.hiddenLayers = hiddenLayers;
            for (int i = 1; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].previousLayer = hiddenLayers[i - 1];
            }
            this.outputLayer = outputLayer;
            if (hiddenLayers.Length > 0)
            {
                outputLayer.previousLayer = hiddenLayers[hiddenLayers.Length - 1];
            }
        }

        public double[] feedForward(double[] inputs)
        {
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                inputs = hiddenLayers[i].feedforward(inputs);
            }
            return outputLayer.feedforward(inputs);
        }

        /// <summary>
        /// An optimised feed-forward intended for multiple input sets where some of their inputs are equal. 
        /// Precomputes the first hidden layer using the common inputs to re-use for each unique input set.
        /// This implementation requires all of the common inputs to come before the unique inputs (i.e., the inputs have the same order as they would during regular feed-forward).
        /// </summary>
        /// <param name="equalInputs">The inputs that each input set has in common.</param>
        /// <param name="uniqueInputs">The set of inputs that are unique to each input set.</param>
        /// <returns></returns>
        virtual public double[,] optimisedFeedForward(double[] equalInputs, double[,] uniqueInputs)
        {
            if (hiddenLayers.Length == 0)
            {
                throw new Exception("Optimised feed-forward requires a hidden layer.");
            }

            //precompute first hidden layer
            HiddenLayer layer = hiddenLayers[0];
            double[] layerNodes = new double[layer.weights.GetLength(1)];
            for (int i = 0; i < layer.weights.GetLength(1); i++)
            {
                for (int j = 0; j < equalInputs.Length; j++)
                {
                    layerNodes[i] += equalInputs[j] * layer.weights[j, i];
                    //Console.WriteLine($"adding {previousLayer[j]} * {weights[j, i]} to {nodes[i]}");
                }
                //Console.WriteLine(nodes[i]);
            }

            double[] layer1NodesBackup = layerNodes;
            int extraInputWeightIndex = equalInputs.Length;
            double[,] results = new double[uniqueInputs.GetLength(0), outputLayer.weights.GetLength(1)];

            //foreach unique input set
            for (int i = 0; i < uniqueInputs.GetLength(0); i++)
            {
                layerNodes = layer1NodesBackup;
                //finish computing first layer
                for (int j = 0; j < layerNodes.Length; j++)
                {
                    for (int k = 0; k < uniqueInputs.GetLength(1); k++)
                    {
                        layerNodes[j] += uniqueInputs[i, k] * layer.weights[k + extraInputWeightIndex, j];
                        //Console.WriteLine($"adding {equalInputs[k]} * {layer.weights[k + extraInputWeightIndex, j]} to {layerNodes[j]}");
                    }
                    layerNodes[j] = layer.activationFunction(layerNodes[j]);
                }

                //compute other layers as normal
                for (int j = 1; j < hiddenLayers.Length; j++)
                {
                    layerNodes = feedForward(layerNodes);
                }
                layerNodes = outputLayer.feedforward(layerNodes);

                for (int j = 0; j < results.GetLength(1); j++)
                {
                    results[i, j] = layerNodes[j];
                }
            }

            return results;
        }

        virtual public void backPropagate(double[] inputs, double[] actualValues, double learningRateMulti, bool printLoss = false)
        {
            double[] ff = feedForward(inputs);
            double[] losses = new double[ff.Length];
            for (int i = 0; i < losses.Length; i++)
            {
                losses[i] = ff[i] - actualValues[i];
            }
            if (printLoss)
            {
                Console.WriteLine(losses[0]);
            }
            outputLayer.backPropagate(losses, learningRateMulti);
        }
    }
}
