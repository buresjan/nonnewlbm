SetFactory("OpenCASCADE");

// Characteristic mesh length
h = DEFINE_H;
Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;

// Cylinder dimensions
LOWER_LENGTH = 30.0 / 100.0;
LOWER_RADIUS = 4.5 / 100.0;
UPPER_LENGTH = 30.0 / 100.0;
UPPER_RADIUS = 4.5 / 100.0;
MIDDLE_LENGTH = 120.0 / 100.0;
MIDDLE_RADIUS = 6.5 / 100.0;

// Offset for positioning along the X-axis
OFFSET = DEFINE_OFFSET;
//RES = DEFINE_RESOLUTION;

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

