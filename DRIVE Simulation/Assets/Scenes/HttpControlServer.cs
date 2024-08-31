// Author: Felix Kirch
// Date: 08.08.2024
// Description: This script receives JSON files containing speed and steering inputs, which are used to control the car.
//              The JSON data is posted by the client to the IP address of the simulation

using System;
using System.Net;
using System.Text;
using System.Threading;
using UnityEngine;

[System.Serializable]
public class ControlValues
{
    // Class to hold speed and angle values received from the JSON input
    public float speed;  // Speed control input
    public float angle;  // Steering control input
}

public class HttpControlServer : MonoBehaviour
{
    private HttpListener listener;  // HTTP listener to accept incoming connections
    private Thread listenerThread;  // Thread to handle the listening in the background
    public int port = 8000;  // Port to listen on
    public CustomCarController carController;

    void Start()
    {
        // Setting up the HttpListener
        listener = new HttpListener();
        listener.Prefixes.Add($"http://*:{port}/");  // Adding the prefix for HTTP connections
        listener.Start();  // Start the HttpListener

        // Starting the listener thread
        listenerThread = new Thread(StartListener);
        listenerThread.IsBackground = true;  // Set the thread as a background thread
        listenerThread.Start();

        Debug.Log("HTTP Server gestartet");
    }

    void OnApplicationQuit()
    {
        // Stop the listener when the application quits
        listener.Stop();
        listenerThread.Abort();  // Abort the listener thread
    }

    private void StartListener()
    {
        while (listener.IsListening)  // Keep running while the listener is active
        {
            try
            {
                // Get the context (i.e., incoming request)
                var context = listener.GetContext();
                // Process the incoming request
                ProcessRequest(context);
            }
            catch (Exception ex)
            {
                // Log any exceptions
                Debug.Log($"HTTP Server Fehler: {ex.Message}");
            }
        }
    }

    private void ProcessRequest(HttpListenerContext context)
    {
        if (context.Request.HttpMethod == "POST")  // Ensure the request method is POST as the third party client posts the JSON data to the simulation
        {
            // Read the request body
            using (var reader = new System.IO.StreamReader(context.Request.InputStream, context.Request.ContentEncoding))
            {
                var json = reader.ReadToEnd();  // Read the full JSON content
                var controlValues = JsonUtility.FromJson<ControlValues>(json);  // Deserialize the JSON into ControlValues object
                carController.SetControlValues(controlValues.speed, controlValues.angle);  // Set the car control values
                Debug.Log($"Empfangen: Speed={controlValues.speed}, Steering={controlValues.angle}");

                // Send an OK response back to the client
                var responseString = "OK";
                var buffer = Encoding.UTF8.GetBytes(responseString);

                context.Response.ContentLength64 = buffer.Length;
                var responseOutput = context.Response.OutputStream;
                responseOutput.Write(buffer, 0, buffer.Length);  // Write the response
                responseOutput.Close();  // Close the response output stream
            }
        }
    }
}