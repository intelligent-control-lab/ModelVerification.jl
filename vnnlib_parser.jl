function read_statements(vnnlib_filename)
    lines = readlines(vnnlib_filename)
    lines = [strip(line) for line in lines]
    @assert length(lines) > 0
    
    # combine lines if case a single command spans multiple lines
    open_parentheses = 0
    statements = []
    current_statement = ""
    
    
    for line in lines
        comment_index = findfirst(isequal(';'), line)
        isnothing(comment_index) || (line = strip(line[1:comment_index-1]))
        isempty(line) && continue
        
        new_open = sum([1 for i = eachmatch(r"\(", line)])
        new_close = sum([1 for i = eachmatch(r"\)", line)])
        open_parentheses += new_open - new_close

        @assert open_parentheses >= 0
        # add space
        current_statement *= isempty(current_statement) ? "" : " "
        current_statement *= line
    
        if open_parentheses == 0
            push!(statements, current_statement)
            current_statement = ""
        end
    end

    isempty(current_statement) || push!(statements, current_statement)
    statements = [join(split(s), " ") for s in statements]
    
    # remove space after '('
    statements = [replace(s, "( " => "(") for s in statements]

    # remove space after ')'
    statements = [replace(s, ") " => ")") for s in statements]
    return statements
end

function update_rv_tuple!(rv_tuple, op, first, second, num_inputs, num_outputs)
    if startswith(first, "X_")
        # Input constraints
        index = parse(Int64, first[3:end])

        @assert !startswith(second, "X") && !startswith(second, "Y")
        @assert 0 <= index < num_inputs

        if op == "<="
            rv_tuple[1][index+1][2] = min(parse(Float64, second), rv_tuple[1][index+1][2])
        else
            rv_tuple[1][index+1][1] = max(parse(Float64, second), rv_tuple[1][index+1][1])
        end

        @assert rv_tuple[1][index+1][1] <= rv_tuple[1][index+1][2]
    else
        # output constraint
        if op == ">="
            # swap order if op is >=
            first, second = second, first
        end

        row = zeros(num_outputs)
        rhs = 0.0

        # assume op is <=
        if startswith(first, "Y_") && startswith(second, "Y_")
            index1 = parse(Int64, first[3:end])
            index2 = parse(Int64, second[3:end])

            row[index1+1] = 1
            row[index2+1] = -1
        elseif startswith(first, "Y_")
            index1 = parse(Int64, first[3:end])
            row[index1+1] = 1
            rhs = parse(Float64, second)
        else
            @assert startswith(second, "Y_")
            index2 = parse(Int64, second[3:end])
            row[index2+1] = -1
            rhs = -1 * parse(Float64, first)
        end

        push!(rv_tuple[2], row)
        push!(rv_tuple[3], rhs)
    end
end

function make_input_box_dict(num_inputs)
    # make a dict for the input box
    rv = Dict(zip(1:num_inputs, [[-Inf,Inf] for i in 1:num_inputs]))
    return rv
end

function read_vnnlib_simple(vnnlib_filename, num_inputs, num_outputs)
    #process in a vnnlib file
    
    # example: "(declare-const X_0 Real)"
    regex_declare = r"^\(declare-const (X|Y)_(\S+) Real\)$"

    # comparison sub-expression
    # example: "(<= Y_0 Y_1)" or "(<= Y_0 10.5)"
    comparison_str = r"\((<=|>=) (\S+) (\S+)\)"

    # example: "(and (<= Y_0 Y_2)(<= Y_1 Y_2))"
    dnf_clause_str = r"\(and (\((<=|>=) (\S+) (\S+)\))+\)"
    
    # example: "(assert (<= Y_0 Y_1))"
    regex_simple_assert = r"^\(assert \((<=|>=) (\S+) (\S+)\)\)$"
    
    # disjunctive-normal-form
    # (assert (or (and (<= Y_3 Y_0)(<= Y_3 Y_1)(<= Y_3 Y_2))(and (<= Y_4 Y_0)(<= Y_4 Y_1)(<= Y_4 Y_2))))
    regex_dnf = r"^\(assert \(or (\(and (\((<=|>=) (\S+) (\S+)\))+\))+\)\)$"

    rv = [] # list of 3-tuples, (box-dict, mat, rhs)
    push!(rv, (make_input_box_dict(num_inputs), [], []))
    
    lines = read_statements(vnnlib_filename)
    
    for line in lines
        #print(f"Line: {line}")

        if length(findall(regex_declare, line)) > 0
            continue
        end

        groups = match(regex_simple_assert, line)
        if !isnothing(groups)
            
            op, first, second = groups.captures
            for rv_tuple in rv
                update_rv_tuple!(rv_tuple, op, first, second, num_inputs, num_outputs)
            end
            
            continue
        end

        ################

        groups = match(regex_simple_assert, line)
        @assert isnothing(groups)

        line = replace(line, "(" => " ")
        line = replace(line, ")" => " ")
        tokens = split(line)
        tokens = tokens[3:end] # skip 'assert' and 'or'

        conjuncts = split(join(tokens, " "), "and")[2:end]

        old_rv = rv
        rv = []
        for rv_tuple in old_rv
            for c in conjuncts
                rv_tuple_copy = deepcopy(rv_tuple)
                push!(rv, rv_tuple_copy)

                c_tokens = [s for s in split(c, " ") if length(s) > 0]

                count = length(c_tokens) ÷ 3

                for i in 1:count
                    op, first, second = c_tokens[3*i-2:3*i]
                    update_rv_tuple!(rv_tuple_copy, op, first, second, num_inputs, num_outputs)
                end
            end
        end
    end
    # merge elements of rv with the same input spec
    merged_rv = Dict()
    
    for rv_tuple in rv
        boxdict = rv_tuple[1]
        matrhs = (rv_tuple[2], rv_tuple[3])

        key = string(boxdict) # merge based on string representation of input box... accurate enough for now

        if haskey(merged_rv, key)
            push!(merged_rv[key][2], matrhs)
        else
            merged_rv[key] = (boxdict, [matrhs])
        end
    end

    # finalize objects (convert dicts to lists and lists to np.array)
    final_rv = []

    for rv_tuple in values(merged_rv)
        box_dict = rv_tuple[1]
        
        box = []

        for d in 1:num_inputs
            r = box_dict[d]

            @assert r[1] != -Inf && r[2] != Inf
            push!(box, r)
        end
        
        spec_list = []

        for matrhs in rv_tuple[2]
            mat = matrhs[1]
            rhs = matrhs[2]
            push!(spec_list, (mat, rhs))
        end

        push!(final_rv, (box, spec_list))
    end

    return final_rv

end

