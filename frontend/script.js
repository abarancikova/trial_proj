const form = document.getElementById('argument-form');
const argumentInput = document.getElementById('argument-input');
const qualityScoreDiv = document.getElementById('quality-score');

// event listener for form submission
form.addEventListener('submit', async (event) => {
    event.preventDefault();
    
    const argumentText = argumentInput.value.trim(); // get the input 
    
    // check if the argument is not empty
    if (argumentText !== '') {
        try {
            // send the argument to the backend
            const response = await fetch('http://localhost:8000', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ argument: argumentText })
            });

            const data = await response.json();

            // display the quality score of the argument
            qualityScoreDiv.textContent = `Quality Score: ${data.quality_score}`;
        } catch (error) {
            console.error('Error:', error);
            // error message
            qualityScoreDiv.textContent = 'There was an error processing your argument. Please try again.';
        }
    } else {
        // print a message if the input is empty
        qualityScoreDiv.textContent = 'Please enter an argument.';
    }
});
