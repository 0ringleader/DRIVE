using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

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
    public bool resetOnOffRoad = true;
    public bool autoSwitchScenes = true;
    public bool randomizeStartingDirection = true;

    private float currentSplineT = 0.0f; // Fortschritt entlang der Spline
    private UIButtonHandler uiButtonHandler; // Reference to UIButtonHandler

    // Start is called before the first frame update
    private void Start()
    {
        targetSpeed = 0.0f; // Initialisieren Sie die Geschwindigkeit auf 0
        targetSteering = 0.0f; // Initialisieren Sie die Lenkung auf 0
        currentSplineT = 0.0f; // Initialisieren des Fortschritts auf der Spline
        uiButtonHandler = FindObjectOfType<UIButtonHandler>(); // Find the UIButtonHandler in the scene
    }

    // Update is called once per frame
void Update()
{
    if (!isControlledByWebsite)
    {
        // Abfrage des Inputs vom Mausrad
        float mouseWheelInput = Input.GetAxis("Mouse ScrollWheel");
        // Abfrage des Inputs von den Pfeiltasten (oder einem Gamepad)
        float arrowKeyInput = Input.GetAxis("Vertical");

        // Wenn ein Pfeiltasten-Eingang vorliegt, verwende diesen, um die Zielgeschwindigkeit direkt zu setzen
        if (Mathf.Abs(arrowKeyInput) > Mathf.Epsilon)
        {
            targetSpeed = arrowKeyInput * maxSpeed;
        }
        // Wenn das Mausrad benutzt wird, addiere den Wert zu targetSpeed
        else if (Mathf.Abs(mouseWheelInput) > Mathf.Epsilon)
        {
            targetSpeed += mouseWheelInput;
            targetSpeed = Mathf.Clamp(targetSpeed, -maxSpeed, maxSpeed); // Clamping um sicherzustellen, dass targetSpeed im erlaubten Bereich bleibt
        }

        // Die Lenkung wird weiterhin mit den Pfeiltasten gesteuert
        targetSteering = Input.GetAxis("Horizontal") * maxSteering;
    }

    if (Input.GetKeyDown(KeyCode.R))
    {
        ResetCar(); // Aufruf der ResetCar-Methode bei Druck der 'R'-Taste
    }

    UpdateSteering(); // Aktualisierung der Lenkung
    UpdateSpeed();    // Aktualisierung der Geschwindigkeit

    // Bewegung des Autos
    var movement = transform.forward * speed * Time.deltaTime;
    transform.position += movement;

    // Drehung der Vorderräder
    frontWheelLeft.localEulerAngles = new Vector3(0, 0, steering + 90);
    frontWheelRight.localEulerAngles = new Vector3(0, 0, steering + 90);

    // Berechnung des Rotationspunkts
    var rotationPoint = new Vector3(0, 0, -0.0675f);

    // Berechnung des Lenkwinkels
    var steeringAngle = steering / wheelbase;

    // Drehung des Autos um den Rotationspunkt
    transform.RotateAround(transform.TransformPoint(rotationPoint), Vector3.up,
        steeringAngle * speed * Time.deltaTime);

    // Begrenzung der Lenkung
    steering = Mathf.Clamp(steering, -maxSteering, maxSteering);

    // Begrenzung der Geschwindigkeit
    speed = Mathf.Clamp(speed, -maxSpeed, maxSpeed);

    // Überprüfung, ob das Auto unter y = -1 ist und falls notwendig zurücksetzen
    if (transform.position.y < -1)
    {
        ResetCar();
    }

    // Aktualisierung der UI-Elemente
    if (uiButtonHandler != null)
    {
        float mappedSpeed = MapValue(targetSpeed, -maxSpeed, maxSpeed, -100, 100);
        float mappedSteering = MapValue(targetSteering, -maxSteering, maxSteering, -100, 100);
        uiButtonHandler.UpdateSpeedText(speed, mappedSpeed);
        uiButtonHandler.UpdateSteeringText(steering, mappedSteering);
    }
}


    public void ResetCar()
    {
        transform.position = new Vector3(0, -0.199f, 0); // Reset position to (0,-0.199,0)
        speed = 0.0f; // Reset speed
        steering = 0.0f; // Reset steering
        //targetSpeed = 0.0f; // Reset target speed
        //targetSteering = 0.0f; // Reset target steering
        if (randomizeStartingDirection && UnityEngine.Random.value > 0.5f)
        {
            transform.rotation = Quaternion.Euler(0, 180, 0); // Randomize rotation
        }
        else
        {
            transform.rotation = Quaternion.identity; // Reset rotation
        }
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
        // Bestimme, ob das Auto beschleunigt oder bremst
        bool isAccelerating = (targetSpeed > speed && speed >= 0) || (targetSpeed < speed && speed <= 0);
    
        // Verwende die Beschleunigungsrate, wenn das Auto beschleunigt, andernfalls die Bremsrate
        float rate = isAccelerating ? accelerationRate : brakingRate;

        // Aktualisiere die Geschwindigkeit mit der korrekten Rate
        speed = Mathf.MoveTowards(speed, targetSpeed, rate * Time.deltaTime);
    }

}