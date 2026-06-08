// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

/**
 * AttentionShift - Calculate the attention shift according to gradient descent on error.
 * @brief Computes the attention shift based on the connection weights, input vector, output vector, error vector, and attention vector.
 * @param weights mat. The connection weights matrix.
 * @param input rowvec. The input vector representing the currently active nodes.
 * @param error colvec. The error vector representing the prediction error.
 * @param gain rowvec. The attention vector representing the current attention gains.
 * @param pnorm double. The p-norm value used for normalization.
 * @param P double. The exponent used for p-norm normalization.
 * @param rho double. The scaling factor for the salience calculation.
 * @return rowvec. The salience vector representing the attention shift for each input node.
 */
// [[Rcpp::export]]
arma::rowvec AttentionShift(arma::mat weights, arma::colvec predictions, arma::rowvec input,
                            arma::colvec error, arma::rowvec gain, double pnorm, double P, double rho)
{
    rowvec salience(gain);                                                // salience vector
    mat attended_activations(weights.n_rows, weights.n_cols, fill::ones); // creaate a matrix to hold intermediate computations

    // calculate model predictions
    mat activations = weights.each_row() % input; // apply attention gain to weights

    // raise salience to power of P-1
    for (uword j = 0; j < gain.n_elem; ++j)
    {
        salience[j] = pow(gain[j], P - 1) * input[j];
    }

    // calculate the attended activations
    attended_activations.each_col() % predictions; // modulate by prediction error
    attended_activations.each_row() % salience;    // modulate by input
    attended_activations.diag().zeros();           // set diagonal to 0

    mat out = (activations - attended_activations); // calculate the difference between activations and attended activations
    out = out.each_col() % error;                   // modulate by the error
    mat out_sum = sum(out, 0);                      // sum across rows
    // TODO: mute stimuli not present
    mat delta_eta = arma::tanh(rho * pow(pnorm, -1) * out_sum % input); // calculate the salience update

    return delta_eta; // calculate the error for each input node
}
