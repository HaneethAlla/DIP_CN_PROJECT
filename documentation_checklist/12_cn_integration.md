

## Overview

The CN component is designed as a client–server extension of the WBC classification system. The image-processing and classification model provides the application logic, while the network layer enables communication between the image-sending client and the classification server.

## Client Side

- The client selects or uploads a WBC image.
    
- The client sends the image to the server as an application-layer request.
    
- The client waits for the classification response.
    
- The received WBC class is displayed to the user.
    

## Server Side

- The server receives the image request.
    
- The server performs the required preprocessing.
    
- The trained 13-class WBC classifier processes the image.
    
- The server returns the predicted class and, optionally, confidence information.
    

## CN Concepts Demonstrated

|CN Concept|Proposed Implementation|Purpose|
|---|---|---|
|Client–server model|Client uploads image; server classifies it|Separates interface from processing|
|Application layer|HTTP request/response API|Transfers image and prediction|
|Transport layer|TCP carries the HTTP connection|Reliable delivery of image data|
|Internet layer|IP addressing between client and server|Logical host-to-host delivery|
|Performance|Measure latency and throughput|Analyze network performance|

## Proposed Communication Sequence

1. Client establishes communication with the server.
    
2. Client sends an HTTP request containing the WBC image.
    
3. TCP provides reliable transport for the HTTP data.
    
4. Server receives and decodes the image.
    
5. Server runs the WBC classification pipeline.
    
6. Server sends the predicted class in an HTTP response.
    
7. Client displays the classification result.
    
8. Optionally, record network latency, response size and throughput.
    

## Status

This section describes the proposed architecture. Update it with implementation details and test results after the client–server module has actually been developed and tested.