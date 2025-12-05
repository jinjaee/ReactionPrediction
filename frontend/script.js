// --- 1. SELECTORS ---
const r1Box = document.getElementById('r1-box');
const r1Name = document.getElementById('r1-name');
const r2Box = document.getElementById('r2-box');
const r2Name = document.getElementById('r2-name');

const productBox = document.getElementById('product-box');
const productName = document.getElementById('product-name');
const yieldText = document.getElementById('yield-text');
const statusMsg = document.getElementById('status-msg');
const otherProductsBox = document.getElementById('other-products-box');

const exampleCells = document.querySelectorAll('.example-cell');

let reactant1 = null;
let reactant2 = null;
let currentStableProducts = [];

// --- 2. CLICK HANDLERS ---
exampleCells.forEach(cell => {
    cell.addEventListener('click', () => {
        const chemKey = cell.getAttribute('data-chem');
        const imgSrc = cell.querySelector('img').src;
        const nameText = cell.querySelector('.example-name').innerText;

        if (!reactant1) {
            reactant1 = chemKey;
            r1Box.innerHTML = `<img src="${imgSrc}" style="width:100%; height:100%; object-fit:contain;">`;
            r1Name.innerText = nameText;
            r1Name.style.color = "#4fffaa";
            statusMsg.innerText = "Select one more reactant...";
        } else if (!reactant2) {
            reactant2 = chemKey;
            r2Box.innerHTML = `<img src="${imgSrc}" style="width:100%; height:100%; object-fit:contain;">`;
            r2Name.innerText = nameText;
            r2Name.style.color = "#4fffaa";

            predictReaction();
        } else {
            resetSimulation();
            reactant1 = chemKey;
            r1Box.innerHTML = `<img src="${imgSrc}" style="width:100%; height:100%; object-fit:contain;">`;
            r1Name.innerText = nameText;
            r1Name.style.color = "#4fffaa";
        }
    });
});

// --- 3. RENDER ENGINE ---
function renderScenario(selectedProduct) {
    // A. Main Product
    const formula = selectedProduct.formula;
    const mainImg = `https://placehold.co/150x150/8A2BE2/white?text=${formula}`;

    productBox.innerHTML = `<img src="${mainImg}" style="width:100%; height:100%; object-fit:contain; border-radius: 8px;">`;
    productName.innerText = formula;
    productName.style.color = "#c58bff";

    // Fake Yield
    const energy = Math.abs(selectedProduct.energy_per_atom);
    const fakeYield = Math.min(100, Math.floor(energy * 40 + 50));
    yieldText.innerText = `${fakeYield}%`;

    // Animation
    productBox.style.opacity = "0.5";
    setTimeout(() => productBox.style.opacity = "1", 150);

    // B. Other Products List
    otherProductsBox.innerHTML = '';

    if (currentStableProducts.length > 1) {
        currentStableProducts.forEach((prod) => {
            // Filter: Don't show the one currently in the Main Box
            if (prod.formula !== selectedProduct.formula) {

                const sideFormula = prod.formula;
                const smallImg = `https://placehold.co/80x80/555555/white?text=${sideFormula}`;

                const card = document.createElement('div');
                card.style.cssText = "display: flex; flex-direction: column; align-items: center; margin: 0 10px; cursor: pointer; transition: transform 0.2s;";
                card.innerHTML = `
                    <img src="${smallImg}" style="width: 60px; height: 60px; border-radius: 6px; border: 1px solid #777;">
                    <span style="color: #ccc; font-size: 0.8rem; margin-top: 5px;">${sideFormula}</span>
                `;

                card.onmouseover = () => card.style.transform = "scale(1.1)";
                card.onmouseout = () => card.style.transform = "scale(1.0)";

                // Swap logic
                card.onclick = () => {
                    renderScenario(prod);
                    statusMsg.innerText = `Selected Product: ${sideFormula}`;
                };

                otherProductsBox.appendChild(card);
            }
        });
    } else {
        otherProductsBox.innerHTML = '<span style="color: #555; font-size: 0.9rem;">No side products.</span>';
    }
}

// --- 4. AI LOGIC WITH FILTERING ---
async function predictReaction() {
    statusMsg.innerText = "Consulting AI Engine...";

    // Clean inputs (e.g. "Cl2" -> "Cl")
    const cleanA = reactant1.replace(/[0-9]/g, '');
    const cleanB = reactant2.replace(/[0-9]/g, '');

    try {
        const response = await fetch('http://localhost:8000/predict_reaction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ element_a: cleanA, element_b: cleanB })
        });

        if (!response.ok) throw new Error(`Server Error: ${response.status}`);

        const data = await response.json();
        console.log("AI Response:", data);

        if (data.stable_products && data.stable_products.length > 0) {

            // --- NEW FILTER LOGIC ---
            // 1. Identify pure elements using Regex (e.g. ^O\d*$ matches "O", "O2", "O3")
            const regexA = new RegExp(`^${cleanA}\\d*$`);
            const regexB = new RegExp(`^${cleanB}\\d*$`);

            // 2. Separate products into Compounds vs Pure Elements
            const compounds = data.stable_products.filter(p =>
                !regexA.test(p.formula) && !regexB.test(p.formula)
            );

            // 3. Decide what to show
            // If compounds exist (e.g. MgO), only show them. 
            // If ONLY pure elements exist (e.g. Mg + K -> Mg, K), show them.
            if (compounds.length > 0) {
                currentStableProducts = compounds;
            } else {
                currentStableProducts = data.stable_products;
            }

            // Start with the first one in our filtered list
            renderScenario(currentStableProducts[0]);
            statusMsg.innerText = "Reaction Successful!";

        } else {
            handleNoReaction();
        }

    } catch (error) {
        console.error("Connection Failed:", error);
        handleError();
    }
}

function handleNoReaction() {
    productBox.innerText = "X";
    productName.innerText = "No Reaction";
    productName.style.color = "#aaa";
    yieldText.innerText = "0%";
    otherProductsBox.innerHTML = "";
    statusMsg.innerText = "These elements do not react.";
}

function handleError() {
    productBox.innerText = "!";
    productName.innerText = "Error";
    productName.style.color = "red";
    statusMsg.innerText = "Check Python Backend (Terminal 1)";
}

function resetSimulation() {
    reactant1 = null;
    reactant2 = null;
    currentStableProducts = [];

    r1Box.innerHTML = "";
    r2Box.innerHTML = "";
    productBox.innerHTML = "?";
    otherProductsBox.innerHTML = "";

    r1Name.innerText = "Empty";
    r2Name.innerText = "Empty";
    productName.innerText = "Unknown";

    r1Name.style.color = "var(--text-color)";
    r2Name.style.color = "var(--text-color)";
    productName.style.color = "var(--text-color)";

    yieldText.innerText = "0%";
    statusMsg.innerText = "Waiting for input...";
}