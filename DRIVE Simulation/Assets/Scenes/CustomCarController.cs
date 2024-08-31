using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

public class CustomCarController : MonoBehaviour
{
    // Ccar control
    public float targetSpeed; // desired speed (manipulated by JSON data)
    public float targetSteering; //desiered steering value
    public Transform frontWheelLeft;
    public Transform frontWheelRight;

    // Constants for physical car dynamics
    private readonly float maxSpeed = 0.8f;
    private readonly float maxSteering = 19.0f;
    private readonly float wheelbase = 0.1375f;
    public float steeringMotorSpeed = 60.0f;
    public float accelerationRate = 10.0f;
    public float brakingRate = 50.0f;

    // Initial state of the car
    private float steering = 0.0f;
    private float speed = 0.0f;

    // Configuration
    public bool isControlledByWebsite = true;
    public bool resetOnOffRoad = true;
    public bool autoSwitchScenes = true;
    public bool randomizeStartingDirection = true;


    private UIButtonHandler uiButtonHandler;

    // Initialization
    private void Start()
    {
        targetSpeed = 0.0f; 
        targetSteering = 0.0f; 
        uiButtonHandler = FindObjectOfType<UIButtonHandler>(); // Find the UIButtonHandler in the scene
    }

    // Update is called once per frame
    private void Update()
    {
        if (!isControlledByWebsite)
        {
            // Handle input for speed and steering
            float mouseWheelInput = Input.GetAxis("Mouse ScrollWheel");
            float arrowKeyInput = Input.GetAxis("Vertical");

            // Direct speed control with arrow keys
            if (Mathf.Abs(arrowKeyInput) > Mathf.Epsilon)
            {
                targetSpeed = arrowKeyInput * maxSpeed;
            }
            // Adjust speed incrementally with mouse wheel
            else if (Mathf.Abs(mouseWheelInput) > Mathf.Epsilon)
            {
                targetSpeed += mouseWheelInput;
                targetSpeed = Mathf.Clamp(targetSpeed, -maxSpeed, maxSpeed);
            }

            // Steering is controlled with the horizontal axis (arrow keys or gamepad)
            targetSteering = Input.GetAxis("Horizontal") * maxSteering;
        }

        // Reset car position and state on 'R' key press
        if (Input.GetKeyDown(KeyCode.R))
        {
            ResetCar();
        }

        // Update steering and speed
        UpdateSteering();
        UpdateSpeed();

        // Move the car forward
        var movement = transform.forward * speed * Time.deltaTime;
        transform.position += movement;

        // Rotate front wheel models
        frontWheelLeft.localEulerAngles = new Vector3(0, 0, steering + 90);
        frontWheelRight.localEulerAngles = new Vector3(0, 0, steering + 90);

        // Calculate steering angle and rotate car around the rotation point
        var rotationPoint = new Vector3(0, 0, -0.0675f);
        var steeringAngle = steering / wheelbase;
        transform.RotateAround(transform.TransformPoint(rotationPoint), Vector3.up, steeringAngle * speed * Time.deltaTime);

        // Clamp steering and speed to their maximum values
        steering = Mathf.Clamp(steering, -maxSteering, maxSteering);
        speed = Mathf.Clamp(speed, -maxSpeed, maxSpeed);

        // Reset the car if it falls off the ground plane
        if (transform.position.y < -1)
        {
            ResetCar();
        }

        // Update UI elements
        if (uiButtonHandler != null)
        {
            float mappedSpeed = MapValue(targetSpeed, -maxSpeed, maxSpeed, -100, 100);
            float mappedSteering = MapValue(targetSteering, -maxSteering, maxSteering, -100, 100);
            uiButtonHandler.UpdateSpeedText(speed, mappedSpeed);
            uiButtonHandler.UpdateSteeringText(steering, mappedSteering);
        }
    }

    // Resets the car to its initial position and state
    public void ResetCar()
    {
        transform.position = new Vector3(0, -0.199f, 0); 
        speed = 0.0f; 
        steering = 0.0f; 

        // Randomize starting direction (circuit direction)
        if (randomizeStartingDirection && UnityEngine.Random.value > 0.5f)
        {
            transform.rotation = Quaternion.Euler(0, 180, 0); 
        }
        else
        {
            transform.rotation = Quaternion.identity; 
        }
    }

    // Set control values from an external source
    public void SetControlValues(float speed, float steering)
    {
        if (isControlledByWebsite)
        {
            Debug.Log($"Setting control values - Speed: {speed}, Steering: {steering}");
            targetSpeed = MapValue(speed, -100, 100, -maxSpeed, maxSpeed);
            targetSteering = MapValue(steering, -100, 100, -maxSteering, maxSteering);
        }
    }

    // Maps control values to physical values
    float MapValue(float value, float fromSource, float toSource, float fromTarget, float toTarget)
    {
        return (value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget;
    }

    // Smoothly updates the steering angle
    void UpdateSteering()
    {
        steering = Mathf.MoveTowards(steering, targetSteering, steeringMotorSpeed * Time.deltaTime);
    }

    // Smoothly updates the speed, considering acceleration or braking
    void UpdateSpeed()
    {
        bool isAccelerating = (targetSpeed > speed && speed >= 0) || (targetSpeed < speed && speed <= 0);
        float rate = isAccelerating ? accelerationRate : brakingRate;
        speed = Mathf.MoveTowards(speed, targetSpeed, rate * Time.deltaTime);
    }
}
