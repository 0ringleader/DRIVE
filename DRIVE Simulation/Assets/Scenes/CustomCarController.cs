using UnityEngine;

public class CustomCarController : MonoBehaviour
{
    public float speed; // Geschwindigkeitsvariable
    public float steering; // Lenkvariable
    public Transform frontWheelLeft;
    public Transform frontWheelRight;
    private readonly float maxSpeed = 10.0f;
    private readonly float maxSteering = 30.0f;
    private readonly float wheelbase = 0.1375f;

    // Start is called before the first frame update
    private void Start()
    {
        speed = 0.0f; // Initialisieren Sie die Geschwindigkeit auf 0
        steering = 0.0f; // Initialisieren Sie die Lenkung auf 0
    }

    // Update is called once per frame
    void Update()
    {
        // Aktualisieren Sie die Geschwindigkeitsvariable mit dem Mausrad
        speed += Input.GetAxis("Mouse ScrollWheel");

        // Begrenzen Sie die Geschwindigkeit auf einen bestimmten Bereich, wenn nötig
        speed = Mathf.Clamp(speed, -maxSpeed, maxSpeed);

        // Aktualisieren Sie die Lenkvariable mit den Pfeiltasten
        steering = Input.GetAxis("Horizontal")*30;

        // Begrenzen Sie die Lenkung auf einen bestimmten Bereich, wenn nötig
        steering = Mathf.Clamp(steering, -maxSteering, maxSteering);

        // Berechnung des Rotationspunkts
        var rotationPoint = new Vector3(0, 0, -0.0675f);

        // Berechnung des Lenkwinkels
        var steeringAngle = steering / wheelbase;

        // Drehung des Autos um den Rotationspunkt
        transform.RotateAround(transform.TransformPoint(rotationPoint), Vector3.up,
            steeringAngle * speed * Time.deltaTime);

        // Fortbewegung des Autos
        var movement = transform.forward * speed * Time.deltaTime;
        transform.position += movement;

        //Drehung der Vorderräder
        frontWheelLeft.localEulerAngles = new Vector3(0, 0, steering + 90);
        frontWheelRight.localEulerAngles = new Vector3(0, 0, steering + 90);
    }
}