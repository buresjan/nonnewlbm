SetFactory("OpenCASCADE");

// 1 optimisation parameter - offset of the bottom cylinder
// upper inflow should be 0.4 ms-1
// lower inflow should be 0.5 ms-1

// Characteristic mesh length
h = DEFINE_H;  // Suggested characteristic mesh length for balanced detail and computation cost
Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;

// Cylinder dimensions
LOWER_LENGTH = 0.05;  // Length of the lower (bottom) cylinder in meters
LOWER_RADIUS = 0.007; // Radius of the lower cylinder in meters
UPPER_LENGTH = 0.05;  // Length of the upper (top) cylinder in meters
UPPER_RADIUS = 0.007; // Radius of the upper cylinder in meters
MIDDLE_LENGTH = 0.17; // Length of the middle (horizontal) cylinder in meters
MIDDLE_RADIUS = 0.011;  // Radius of the middle cylinder in meters

// Offset for positioning along the X-axis - bounds -0.02 to +0.02
OFFSET = DEFINE_OFFSET;  // Adjust if needed for positioning; set to 0 for centered positioning

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// First Cylinder - Lower /////////////////////////////////////////////
////////////////////////////////////////////// ID - 00001 ////////////////////////////////////////////////////////

// Define points along the axis of the lower cylinder
Point(1) = {OFFSET, 0.0, LOWER_LENGTH, h};
Point(2) = {OFFSET, 0.0, 0.0, h};

// Create line and wire for lower cylinder extrusion
Line(1) = {2, 1};
Wire(2) = {1};

// Disk representing the base of the lower cylinder
Disk(1) = {OFFSET, 0.0, LOWER_LENGTH, LOWER_RADIUS};

// Extrude the surface to form the first cylinder volume
Extrude { Surface{1}; } Using Wire {2}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// Second Cylinder - Upper ////////////////////////////////////////////
////////////////////////////////////////////// ID - 00101 ////////////////////////////////////////////////////////

// Define points along the axis of the upper cylinder
Point(101) = {0.0, 0.0, LOWER_LENGTH + UPPER_LENGTH, h};
Point(102) = {0.0, 0.0, LOWER_LENGTH, h};

// Create line and wire for upper cylinder extrusion
Line(101) = {102, 101};
Wire(102) = {101};

// Disk representing the base of the upper cylinder
Disk(101) = {0.0, 0.0, LOWER_LENGTH + UPPER_LENGTH, UPPER_RADIUS};

// Extrude the surface to form the second cylinder volume
Extrude { Surface{101}; } Using Wire {102}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// Third Cylinder - Middle /////////////////////////////////////////////
////////////////////////////////////////////// ID - 10001 ////////////////////////////////////////////////////////

// Define points along the axis of the middle cylinder
Point(10001) = {-MIDDLE_LENGTH / 2.0, 0.0, LOWER_LENGTH, h};
Point(10002) = {MIDDLE_LENGTH / 2.0, 0.0, LOWER_LENGTH, h};

// Create line and wire for middle cylinder extrusion
Line(10001) = {10002, 10001};
Wire(10002) = {10001};

// Disk representing the base of the middle cylinder
Disk(10001) = {-MIDDLE_LENGTH / 2.0, 0.0, LOWER_LENGTH, MIDDLE_RADIUS};

// Orient disk along the XY plane
Rotate{{0, 1, 0}, {-MIDDLE_LENGTH / 2.0, 0.0, LOWER_LENGTH}, Pi/2}{Surface{10001};}

// Extrude the surface to form the third cylinder volume
Extrude { Surface{10001}; } Using Wire {10002}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// Union and Cleanup ///////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Union the three volumes and delete original parts
BooleanUnion{ Volume{1}; Delete; }{ Volume{2, 3}; Delete; }

// Delete any remaining surfaces
Recursive Delete {
  Surface{1, 101, 10001};
}

// Delete any remaining lines
Recursive Delete {
  Line{1, 101, 10001};
}
