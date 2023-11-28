/*
 * =============================================================
 * LI_network.c
 *
 * Approximate analytic (zero-order hold) solution to a LI neural network, specified 
 * by the weight matrix
 *
 * =============================================================
 */

/* Mark Humphries 31/1/2005 */

#include "mex.h"
#include <math.h>

#define enum BOOLEAN {false = 0, true = 1};

extern double output_fcn(int,double,double,double); /* prototypes */
extern double C_sum_array(double*,int,int);
extern void abs_double(double*);

void LI_network(double *weights, 
                double *inputs, 
                double *thresh, 
                double *slope, 
                double *outputs, 
                double *act, 
                double *steps, 
                double *sample_out, 
                int max_steps, 
                double stopping,
                double* decay,
                int N, 
                int pulse, 
                double* sample_points,
                double* p_out,
                double* p_act)
{
  double *act_old, *delta_a, *converged, *old_outputs, *i_decay;
  double u, sum_con = 0, temp = 0;
  int sample_count, loop, in, neuron, idx;
  
  *steps = 0;
  u = 0;
  sample_count = 0;

  /* create local arrays - assign memory */
  i_decay = mxCalloc(N+1,sizeof(double));
  old_outputs = mxCalloc(N+1,sizeof(double));
  act_old = mxCalloc(N+1,sizeof(double));
  delta_a = mxCalloc(N+1,sizeof(double));
  converged = mxCalloc(N+1,sizeof(double));
	  
  /* create variables here for efficency and initialise others */
  /* copy in old activations and outputs to re-initialise network from previous run */
  for (neuron = 0;neuron < N;neuron++){
  	i_decay[neuron] = 1 - decay[neuron];
  	outputs[neuron] = p_out[neuron];
  	act[neuron] = p_act[neuron];
  	old_outputs[neuron] = p_out[neuron];
  	act_old[neuron] = p_act[neuron];
  	converged[neuron] = 0;
  }
  
  do 
  {
    
    for (neuron= 0; neuron < N; neuron++){
        for(in = 0; in < N; in++){
            /* each column contains inputs to that neuron                   */
            /* and the weight matrix is now an array arranged by columns    */
            /* so count down the array                                      */    
            
            /* REWRITE THIS USING POINTER ARITHMETIC */
            u += (weights[neuron*N+in] * old_outputs[in]);
            /*u += (*(weights+(neuron*N+in)) * *(old_outputs+in));*/ /* This works but isn't particulary more efficient */
            
            /* mexPrintf("%i",(int)weights[neuron*N+in]); */
            /* mexPrintf("%f ",u); */
        }
        /*mexPrintf("%f\n",act_old[neuron]); */

        act[neuron] = (inputs[neuron] + u) * i_decay[neuron] + (act_old[neuron] * decay[neuron]);
        u = 0;
        
        delta_a[neuron] = act[neuron] - act_old[neuron];
       
       	/* if input is a pulse then set input vector to zero after first timestep */
       	if (pulse == 1){ inputs[neuron] = 0;}
       	
        /* calculate output  - specify type by first argument */
        outputs[neuron] = output_fcn(1,act[neuron],slope[neuron],thresh[neuron]);
                        
        /* store the outputs if at correct interval */
        if (*steps == sample_points[sample_count])
        {
            idx = neuron+N*sample_count;
            sample_out[idx] = outputs[neuron];
        }
    }   /* finish neuron update loop */
    
    if (*steps == sample_points[sample_count]) sample_count++;    /* increment counter */ 
    
    for (loop = 0; loop < N; loop++) {
        abs_double(&delta_a[loop]);                   /* make sure is absolute value */
        converged[loop] = delta_a[loop] < stopping;    
        /*converged[loop] = old_outputs[loop] - outputs[loop];*/
        old_outputs[loop] = outputs[loop];
        act_old[loop] = act[loop];
    }
    
    sum_con = C_sum_array(converged,N,true);
    if (*steps < 2) sum_con = 0;   /* don't quit on first time-step */
    
    if ((*steps)++ >= max_steps) return;
  }
  /* while(sum_con > stopping); */
  while (sum_con < N);
  
  /* destroy temporary arrays */
  mxFree(old_outputs);
  mxFree(act_old);
  mxFree(delta_a);
  mxFree(i_decay);
  mxFree(converged);
}

/* C-style sum function */
double C_sum_array(double* num_array,int n_rows,int blnAbs)
{
    int i;
    double sum = 0;
    /* return the sum of the absolute values */
    if (blnAbs){
        for (i = 0; i < n_rows; i++){
            abs_double(&num_array[i]);
            sum += num_array[i]; 
            /*mexPrintf("Array entry %f, total %f\n",num_array[i],sum);*/
        }
    }
    else {
        for (i = 0; i < n_rows; i++){
            sum += num_array[i]; 
        }
    }
    return sum;
}

/* C function for returning absolute value of double */
void abs_double(double *num)
{   
    if (*num < 0) *num *= -1;
}

/*
*   OUTPUT function - select appropriate output form
*   specify any values for slope and threshold if meaningless
*   type 1 = ramp; 2 = tanh(); 3 = tanh()H()
*/
double output_fcn(int type,double a,double s,double t){
    double output = 0;
    
    switch(type){
        case 1:
            /* ramp */
            if (a < t) output = 0;
            else if (a > 1/s+t) output = 1;
            else output = s *(a-t);     
            break;
        case 2:
        case 3:
            output = tanh(a);
            break;
    } 
    /*mexPrintf("%f\n",act[neuron]);*/
    if (type == 3) {
        if(output < 0) output = 0;
    } 
    return output;
}

/* The gateway routine */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  double *inputs, *weights, *outputs, *activations, *decay, *steps, *thresh, *slope, *sample_points, *sample_out;
  double *p_out, *p_act;
  double stopping;
  int max_steps,N,pulse,n_samples;
  
  /*  Check for proper number of arguments. */
  /* NOTE: You do not need an else statement when using
     mexErrMsgTxt within an if statement. It will never
     get to the else statement if mexErrMsgTxt is executed.
     (mexErrMsgTxt breaks you out of the MEX-file.) 
  */
  if (nrhs != 13) 
    mexErrMsgTxt("Eleven inputs required."); 
    
  if (nlhs != 4) 
    mexErrMsgTxt("Four outputs required.");
    
  /* Create a pointer to the input matrix y. */
  weights = mxGetPr(prhs[0]);
  inputs = mxGetPr(prhs[1]);
  thresh = mxGetPr(prhs[2]);
  slope = mxGetPr(prhs[3]);
  sample_points = mxGetPr(prhs[4]);
  decay = mxGetPr(prhs[5]);
  
  /* Get the scalar inputs */
  max_steps = mxGetScalar(prhs[6]);
  stopping = mxGetScalar(prhs[7]);
  N = mxGetScalar(prhs[8]);
  pulse = mxGetScalar(prhs[9]);
  n_samples = mxGetScalar(prhs[10]);
  
  p_out = mxGetPr(prhs[11]);
  p_act = mxGetPr(prhs[12]);
    
  /* Set the first output pointer to the output matrix. */
  plhs[0] = mxCreateDoubleMatrix(N,1, mxREAL);
  
  /* Set the second output pointer to an output array */
  plhs[1] = mxCreateDoubleMatrix(1,1, mxREAL);
  
  /* Set the third output pointer to the output matrix. */
  plhs[2] = mxCreateDoubleMatrix(N,1, mxREAL);
  
  /* Set the fourth output pointer to the output matrix. */
  plhs[3] = mxCreateDoubleMatrix(N*n_samples,1, mxREAL);
  
  /* Create a C pointer to a copy of the output matrix. */
  outputs = mxGetPr(plhs[0]);
  steps = mxGetPr(plhs[1]);
  activations = mxGetPr(plhs[2]);
  sample_out = mxGetPr(plhs[3]);
  
  /* Call the C subroutine. */
  LI_network(weights,inputs,thresh,slope,outputs,activations,steps,sample_out,max_steps,stopping,decay,N,pulse,sample_points,p_out,p_act);
}

