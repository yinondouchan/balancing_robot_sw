/*
 * sampleUffSSD.h
 *
 *  Created on: Sep 26, 2019
 *      Author: yinon
 */

#ifndef SAMPLEUFFSSD_H_
#define SAMPLEUFFSSD_H_

#include "BatchStreamPPM.h"
#include "common/common.h"
#include "common/argsParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

static Logger gLogger;

ICudaEngine* ssd_loadModelAndCreateEngine();
void ssd_doInference(IExecutionContext& context, float* inputData, float* detectionOut, int* keepCount, int batchSize);


#endif /* SAMPLEUFFSSD_H_ */
