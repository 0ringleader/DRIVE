using UnityEngine;
using UnityEngine.Serialization;

public class CustomCarController : MonoBehaviour
{
    public float targetSpeed; // Geschwindigkeitsvariable
    public float targetSteering; // Lenkvariable
    public Transform frontWheelLeft;
    public Transform frontWheelRight;
    private readonly float maxSpeed = 0.8f;
    private readonly float maxSteering = 19.0f;
    private readonly float wheelbase = 0.1375f;
    private float steering = 0.0f;
    private float speed = 0.0f;
    public float steeringMotorSpeed = 60.0f;
    public float accelerationRate = 10.0f;
    public float brakingRate = 50.0f;
    public bool isControlledByWebsite = true;

    // Start is called before the first frame update
    private void Start()
    {
        targetSpeed = 0.0f; // Initialisieren Sie die Geschwindigkeit auf 0
        targetSteering = 0.0f; // Initialisieren Sie die Lenkung auf 0
    }

    // Update is called once per frame
void Update()
{
    if (!isControlledByWebsite)
    {
        // Update the speed variable with the mouse wheel
        targetSpeed += Input.GetAxis("Mouse ScrollWheel");

        // Update the steering variable with the arrow keys
        targetSteering = Input.GetAxis("Horizontal") * 30;
    }

    if (Input.GetKeyDown(KeyCode.R))
    {
        ResetCar(); // Call the ResetCar method when 'R' is pressed
    }

    UpdateSteering();
    UpdateSpeed();

    // Car movement
    var movement = transform.forward * speed * Time.deltaTime;
    transform.position += movement;

        //Drehung der Vorderräder
        frontWheelLeft.localEulerAngles = new Vector3(0, 0, steering + 90);
        frontWheelRight.localEulerAngles = new Vector3(0, 0, steering + 90);
        
        // Berechnung des Rotationspunkts
        var rotationPoint = new Vector3(0, 0, -0.0675f);

        // Berechnung des Lenkwinkels
        var steeringAngle = steering / wheelbase;
        
        // Drehung des Autos um den Rotationspunkt
        transform.RotateAround(transform.TransformPoint(rotationPoint), Vector3.up,
            steeringAngle * speed * Time.deltaTime);
        
        // Begrenzen Sie die Lenkung auf einen bestimmten Bereich, wenn nötig
        steering = Mathf.Clamp(steering, -maxSteering, maxSteering);

        // Begrenzen Sie die Geschwindigkeit auf einen bestimmten Bereich, wenn nötig
        speed = Mathf.Clamp(speed, -maxSpeed, maxSpeed);




    }

    private void ResetCar()
{
    transform.position = Vector3.zero; // Reset position to (0,0,0)
    speed = 0.0f; // Reset speed
    steering = 0.0f; // Reset steering
    targetSpeed = 0.0f; // Reset target speed
    targetSteering = 0.0f; // Reset target steering
    transform.rotation = Quaternion.identity; // Reset rotation  
}
    
    public void SetControlValues(float speed, float steering)
    {
        if (isControlledByWebsite)
        {
            Debug.Log($"Setting control values - Speed: {speed}, Steering: {steering}");
            targetSpeed = MapValue(speed, -100, 100, -maxSpeed, maxSpeed);
            targetSteering = MapValue(steering, -100, 100, -maxSteering, maxSteering);

        }
    }
    
    float MapValue(float value, float fromSource, float toSource, float fromTarget, float toTarget)
    {
        return (value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget;
    }
    
    void UpdateSteering()
    {
        steering = Mathf.MoveTowards(steering, targetSteering, steeringMotorSpeed * Time.deltaTime);
    }
    
    void UpdateSpeed()
    {
        float rate = speed < targetSpeed ? accelerationRate : brakingRate;
        speed = Mathf.MoveTowards(speed, targetSpeed, rate * Time.deltaTime);
    }

    
}